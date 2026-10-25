"""Tests for handing each player the streamserver address of its own network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from music_assistant_models.enums import IdentifierType, MediaType
from music_assistant_models.player import PlayerMedia

from music_assistant.controllers.streams.controller import StreamsController

PORT = 8097
# a host on two networks: the primary one first, as the publish candidates rank it
PRIMARY_IP = "192.168.129.10"
SECONDARY_IP = "192.168.128.10"
PUBLISH_CANDIDATES = (PRIMARY_IP, SECONDARY_IP)
PRIMARY_BASE_URL = f"http://{PRIMARY_IP}:{PORT}"
SECONDARY_BASE_URL = f"http://{SECONDARY_IP}:{PORT}"
SECONDARY_PLAYER_IP = "192.168.128.50"
ROUTE_LOOKUP = "music_assistant.controllers.streams.controller.route_source_ip"


def _streams_controller(
    bind_ip: str = "0.0.0.0",
    configured_publish_ip: str | None = None,
    publish_candidates: tuple[str, ...] = PUBLISH_CANDIDATES,
    player_ip: str | None = SECONDARY_PLAYER_IP,
) -> StreamsController:
    """
    Build a streams controller with only its addressing state populated.

    :param bind_ip: Address the streamserver binds to.
    :param configured_publish_ip: The explicitly configured publish IP, or None when auto.
    :param publish_candidates: Host addresses reachable from the local network, ranked.
    :param player_ip: IP address the player under test reports, or None for none.
    """
    mass = MagicMock()
    mass.config.get_raw_core_config_value.return_value = "GLOBAL"
    player = MagicMock()
    player.device_info.identifiers = {IdentifierType.IP_ADDRESS: player_ip} if player_ip else {}
    player.config.get_value.return_value = "flac"
    mass.players.get_player.return_value = player
    mass.player_queues.get.return_value = None
    controller = StreamsController(mass)
    controller._configured_publish_ip = configured_publish_ip
    controller.publish_port = PORT
    controller._resolve_publish_state(bind_ip, publish_candidates)
    return controller


class TestGetBaseUrl:
    """get_base_url picks the address of the interface the device is reached on."""

    def test_device_on_secondary_network_gets_secondary_address(self) -> None:
        """A player on the second network is handed the host address on that network."""
        controller = _streams_controller()
        with patch(ROUTE_LOOKUP, return_value=SECONDARY_IP) as route_lookup:
            assert controller.get_base_url(SECONDARY_PLAYER_IP) == SECONDARY_BASE_URL
        route_lookup.assert_called_once_with(SECONDARY_PLAYER_IP)

    def test_device_on_primary_network_gets_default_address(self) -> None:
        """A player reached through the primary interface keeps the default base URL."""
        controller = _streams_controller()
        with patch(ROUTE_LOOKUP, return_value=PRIMARY_IP):
            assert controller.get_base_url("192.168.129.50") == PRIMARY_BASE_URL

    @pytest.mark.parametrize("routed_ip", ["", "172.17.0.1"])
    def test_unpublished_route_falls_back_to_default(self, routed_ip: str) -> None:
        """
        A route that is unknown, or leaves through an address not published, is not used.

        :param routed_ip: Local address the route lookup resolves to.
        """
        controller = _streams_controller()
        with patch(ROUTE_LOOKUP, return_value=routed_ip):
            assert controller.get_base_url(SECONDARY_PLAYER_IP) == PRIMARY_BASE_URL

    def test_configured_publish_ip_wins_for_every_device(self) -> None:
        """An explicitly configured publish IP is handed to every player, unrouted."""
        controller = _streams_controller(configured_publish_ip="203.0.113.9")
        with patch(ROUTE_LOOKUP) as route_lookup:
            assert controller.get_base_url(SECONDARY_PLAYER_IP) == f"http://203.0.113.9:{PORT}"
        route_lookup.assert_not_called()

    def test_specific_bind_ip_wins_for_every_device(self) -> None:
        """A streamserver pinned to one interface is only reachable on that one."""
        controller = _streams_controller(bind_ip=PRIMARY_IP)
        with patch(ROUTE_LOOKUP) as route_lookup:
            assert controller.get_base_url(SECONDARY_PLAYER_IP) == PRIMARY_BASE_URL
        route_lookup.assert_not_called()

    def test_without_target_returns_default(self) -> None:
        """A URL not meant for one device keeps the default base URL."""
        controller = _streams_controller()
        assert controller.get_base_url() == PRIMARY_BASE_URL

    def test_player_without_ip_gets_default_address(self) -> None:
        """A player that reports no IP address, such as a group, keeps the default."""
        controller = _streams_controller(player_ip=None)
        with patch(ROUTE_LOOKUP) as route_lookup:
            assert controller.get_player_base_url("group_player") == PRIMARY_BASE_URL
        route_lookup.assert_not_called()


class TestOwnStreamUrls:
    """A URL handed out on any published address is recognised as this server's."""

    @pytest.mark.parametrize("base_url", [PRIMARY_BASE_URL, SECONDARY_BASE_URL])
    def test_url_on_any_published_address_is_ours(self, base_url: str) -> None:
        """
        Both the default and the per-network base URL count as this streamserver.

        :param base_url: Base URL the player was handed.
        """
        controller = _streams_controller()
        url = f"{base_url}/flow/session/queue/item/player.flac"
        assert controller.is_stream_url(url)
        assert controller.get_stream_url_path(url) == "/flow/session/queue/item/player.flac"

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            f"http://{PRIMARY_IP}:8095/imageproxy/abc",
            f"http://{PRIMARY_IP}:{PORT}0/flow/x",
            "https://example.com/stream.mp3",
        ],
    )
    def test_foreign_url_is_not_ours(self, url: str | None) -> None:
        """
        Another port, host or scheme is not this streamserver.

        :param url: URL a player could report.
        """
        controller = _streams_controller()
        assert not controller.is_stream_url(url)
        assert controller.get_stream_url_path(url) is None

    def test_base_urls_lists_default_first(self) -> None:
        """Every published address has a base URL, the default one first."""
        controller = _streams_controller()
        assert controller.base_urls == (PRIMARY_BASE_URL, SECONDARY_BASE_URL)


class TestLocalizeUrl:
    """localize_url moves a URL of this server onto the address the player reaches."""

    def test_own_url_is_moved_to_player_network(self) -> None:
        """A URL built on the default base URL is rewritten for a player elsewhere."""
        controller = _streams_controller()
        with patch(ROUTE_LOOKUP, return_value=SECONDARY_IP):
            assert (
                controller.localize_url(f"{PRIMARY_BASE_URL}/imageproxy/abc?size=512", "cast")
                == f"{SECONDARY_BASE_URL}/imageproxy/abc?size=512"
            )

    def test_foreign_url_is_left_alone(self) -> None:
        """A URL of another server is returned as it came in."""
        controller = _streams_controller()
        with patch(ROUTE_LOOKUP, return_value=SECONDARY_IP):
            url = "https://example.com/cover.jpg"
            assert controller.localize_url(url, "cast") == url


class TestResolveStreamUrl:
    """resolve_stream_url hands each player a URL on its own network."""

    async def test_queue_item_url_uses_player_network(self) -> None:
        """A queue item stream URL is built on the address the player reaches."""
        controller = _streams_controller()
        media = PlayerMedia(
            uri="library://track/1",
            media_type=MediaType.TRACK,
            source_id="queue",
            queue_item_id="item",
            queue_session_id="session",
        )
        with patch(ROUTE_LOOKUP, return_value=SECONDARY_IP):
            url = await controller.resolve_stream_url("cast", media)
        assert url.startswith(f"{SECONDARY_BASE_URL}/")
        assert url.endswith("/session/queue/item/cast.flac")

    async def test_prebuilt_url_is_moved_to_player_network(self) -> None:
        """An announcement URL built ahead of time is moved to the player's network."""
        controller = _streams_controller()
        media = PlayerMedia(
            uri=f"{PRIMARY_BASE_URL}/announcement/cast.mp3",
            media_type=MediaType.ANNOUNCEMENT,
        )
        with patch(ROUTE_LOOKUP, return_value=SECONDARY_IP):
            url = await controller.resolve_stream_url("cast", media)
        assert url == f"{SECONDARY_BASE_URL}/announcement/cast.mp3"

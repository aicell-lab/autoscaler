"""Unit tests for ``BioEngineProxyActor._get_accelerator_type``.

Exercises the method against plain resource dicts so no Ray cluster is needed.
"""

from bioengine.cluster.proxy_actor import BioEngineProxyActor

_get_accelerator_type = (
    BioEngineProxyActor.__ray_metadata__.modified_class._get_accelerator_type
)


def test_reads_the_accelerator_type_resource():
    resources = {"CPU": 8.0, "GPU": 1.0, "accelerator_type:A40": 1.0}
    assert _get_accelerator_type(None, resources) == "A40"


def test_type_starting_with_a_prefix_character_survives():
    # str.lstrip("accelerator_type:") strips the character *set*, so a type
    # whose first characters all appear in the prefix loses them.
    resources = {"GPU": 1.0, "accelerator_type:tesla": 1.0}
    assert _get_accelerator_type(None, resources) == "tesla"


def test_returns_none_without_an_accelerator_resource():
    assert _get_accelerator_type(None, {"CPU": 8.0}) is None

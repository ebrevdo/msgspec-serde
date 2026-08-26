from __future__ import annotations

import copy
import importlib
import shutil
import struct
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from types import ModuleType
from typing import Any

import msgspec
import numpy as np
import pytest

import msgspec_flatbuffers._dynamic as dynamic_module
from msgspec_flatbuffers import (
    DynamicModelOverrides,
    DynamicView,
    GenerationError,
    InvalidBufferError,
    TableView,
    compile_schema,
    dec_hook,
    dynamic_types,
    enc_hook,
    generate,
)

SCHEMAS = Path(__file__).parent / "fixtures" / "dynamic"
PAYLOAD_SCHEMA = SCHEMAS / "payload.fbs"
ENVELOPE_SCHEMA = SCHEMAS / "envelope.fbs"
HAS_FLATC = shutil.which("flatc") is not None
METRIC_TAG = "Example.Dynamic.Metric"


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path.remove(value)


def _clear_generated_modules(package: str = "example") -> None:
    for name in tuple(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def generated_modules(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[ModuleType, ModuleType]]:
    if not HAS_FLATC:
        pytest.skip("flatc is not installed")
    output = tmp_path_factory.mktemp("dynamic-generated")
    generate(PAYLOAD_SCHEMA, output, project_root=SCHEMAS)
    generate(ENVELOPE_SCHEMA, output, project_root=SCHEMAS)

    _clear_generated_modules()
    with _temporary_sys_path(output):
        payload = importlib.import_module("example.dynamic.metric")
        envelope = importlib.import_module("example.dynamic.envelope")
        yield payload, envelope
    _clear_generated_modules()


def _model(payload: ModuleType, envelope: ModuleType) -> Any:
    metric = payload.Metric(
        name="latency",
        values=np.array([1.25, 2.5], dtype=np.float32),
    )
    return envelope.Envelope(payload=envelope.EnvelopePayload(metric), note="known")


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_custom_dynamic_attributes_are_reflected() -> None:
    envelope_schema = compile_schema(ENVELOPE_SCHEMA, project_root=SCHEMAS)
    assert envelope_schema.root is not None
    fields = {field.name: field for field in envelope_schema.root.fields}
    assert fields["payload"].dynamic_flatbuffer == "payload_type"
    assert fields["payload"].dynamic_allow == "Example.Dynamic.*"

    payload_schema = compile_schema(PAYLOAD_SCHEMA, project_root=SCHEMAS)
    assert payload_schema.root is not None
    assert any(
        attribute.key == "dynamic_extension"
        for attribute in payload_schema.root.attributes
    )


def test_known_dynamic_value_round_trips_msgspec_formats(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    model = _model(payload, envelope)
    assert envelope.Envelope.__annotations__["payload"] == "EnvelopePayload | None"
    assert "EnvelopePayload" in envelope.__all__
    assert "payload_type" not in envelope.Envelope.__annotations__
    assert payload.Metric.__struct_config__.tag is None
    assert envelope.Envelope.__struct_config__.tag is None

    builtins = msgspec.to_builtins(model, enc_hook=enc_hook)
    metric_value = {
        "name": "latency",
        "values": [1.25, 2.5],
    }
    assert msgspec.to_builtins(model.payload.value, enc_hook=enc_hook) == metric_value
    assert builtins["payload"] == {
        "__msgspec_flatbuffers_type__": METRIC_TAG,
        "value": metric_value,
    }
    assert msgspec.convert(
        builtins,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    ) == model

    encoded = msgspec.json.encode(model, enc_hook=enc_hook)
    decoded = msgspec.json.decode(
        encoded,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    )
    assert decoded == model
    assert type(decoded.payload) is envelope.EnvelopePayload
    assert isinstance(decoded.payload.value, payload.Metric)

    flatbuffer = decoded.to_flatbuffer()
    assert envelope.EnvelopeView.from_buffer(flatbuffer).to_model() == decoded

    disallowed = copy.deepcopy(builtins)
    disallowed["payload"]["__msgspec_flatbuffers_type__"] = "Other.Metric"
    with pytest.raises(msgspec.ValidationError, match="outside"):
        msgspec.convert(
            disallowed,
            type=envelope.Envelope,
            dec_hook=dec_hook,
        )


def test_dynamic_model_subclass_overrides_apply_to_all_decoders(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    namespace = {"Metric": payload.Metric}
    exec(
        """
class ValidatedMetric(Metric, dict=True):
    def __post_init__(self):
        if not self.name:
            raise ValueError("empty metric name")
        self.was_validated = True

class OtherMetric(Metric, dict=True):
    def __post_init__(self):
        self.was_validated = "other"

class InvalidMetric(Metric):
    application_note: str = ""
""",
        namespace,
    )
    validated_metric = namespace["ValidatedMetric"]
    other_metric = namespace["OtherMetric"]
    invalid_metric = namespace["InvalidMetric"]
    overrides = DynamicModelOverrides({payload.Metric: validated_metric})

    decorated = DynamicModelOverrides()

    @decorated.override(payload.Metric)
    class DecoratedMetric(payload.Metric):
        pass

    assert decorated[payload.Metric] is DecoratedMetric

    merged = DynamicModelOverrides() | overrides
    assert isinstance(merged, DynamicModelOverrides)
    assert merged[payload.Metric] is validated_metric
    merged |= {payload.Metric: other_metric}
    assert merged[payload.Metric] is other_metric
    reversed_merge = {payload.Metric: other_metric} | overrides
    assert isinstance(reversed_merge, DynamicModelOverrides)
    assert reversed_merge[payload.Metric] is validated_metric

    model = envelope.Envelope(
        payload=envelope.EnvelopePayload(
            validated_metric(
                name="latency",
                values=np.array([1.25, 2.5], dtype=np.float32),
            )
        )
    )
    flatbuffer = model.to_flatbuffer()
    decoded = envelope.Envelope.from_flatbuffer(
        flatbuffer,
        dynamic_overrides=overrides,
    )
    from_view = envelope.EnvelopeView.from_buffer(flatbuffer).to_model(
        dynamic_overrides=overrides,
    )
    for restored in (decoded, from_view):
        assert restored.payload is not None
        assert type(restored.payload.value) is validated_metric
        assert restored.payload.value.was_validated

    payload_view = envelope.EnvelopeView.from_buffer(flatbuffer).payload
    assert payload_view is not None
    payload_model = payload_view.to_model(dynamic_overrides=overrides)
    assert type(payload_model.value) is validated_metric

    encoded = msgspec.json.encode(model, enc_hook=enc_hook)
    assert b"was_validated" not in encoded
    json_decoded = msgspec.json.decode(
        encoded,
        type=envelope.Envelope,
        dec_hook=overrides.dec_hook,
    )
    assert json_decoded.payload is not None
    assert type(json_decoded.payload.value) is validated_metric
    assert json_decoded.payload.value.was_validated

    default_decoded = msgspec.json.decode(
        encoded,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    )
    assert default_decoded.payload is not None
    assert type(default_decoded.payload.value) is payload.Metric

    with pytest.raises(TypeError, match="serialized msgspec fields"):
        DynamicModelOverrides({payload.Metric: invalid_metric})
    with pytest.raises(TypeError, match="serialized msgspec fields"):
        envelope.EnvelopePayload(
            invalid_metric(
                name="invalid",
                values=np.array([], dtype=np.float32),
            )
        )


def test_known_dynamic_value_round_trips_flatbuffer(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    model = _model(payload, envelope)
    buffer = model.to_flatbuffer()
    view = envelope.EnvelopeView.from_buffer(buffer)
    native_model = envelope.Envelope.from_flatbuffer(buffer)
    assert native_model == model
    assert type(native_model.payload) is envelope.EnvelopePayload
    raw = view.payload_raw
    assert view.payload_type_raw == METRIC_TAG
    assert raw is not None
    assert raw.readonly
    assert raw.obj is buffer.obj

    dynamic = view.payload
    assert isinstance(dynamic, DynamicView)
    assert dynamic is view.payload
    assert dynamic.tag == METRIC_TAG
    value = dynamic.value
    assert isinstance(value, payload.MetricView)
    assert value is dynamic.value
    assert value.name == "latency"
    assert view.to_model() == model


def test_large_dynamic_payload_uses_exact_native_sizing(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    metric = payload.Metric(
        name="large",
        values=np.arange(65_536, dtype=np.float32),
    )
    model = envelope.Envelope(payload=envelope.EnvelopePayload(metric))

    buffer = model.to_flatbuffer()

    assert buffer.obj._allocation_size >= len(buffer)
    assert buffer.obj._allocation_size * 100 <= len(buffer) * 102
    assert envelope.EnvelopeView.from_buffer(buffer).to_model() == model


def test_dynamic_field_rejects_invalid_model_values(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    model = _model(payload, envelope)

    with pytest.raises(TypeError, match="DynamicValue"):
        envelope.Envelope(payload=model.payload.value).to_flatbuffer()

    with pytest.raises(ValueError, match="outside"):
        envelope.Envelope(
            payload=envelope.EnvelopePayload.opaque("Other.Metric", b"data")
        ).to_flatbuffer()


def test_dynamic_view_rejects_invalid_nested_flatbuffer(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    buffer = _model(payload, envelope).to_flatbuffer()
    invalid_payload = bytearray(buffer)
    probe = envelope.EnvelopeView.from_buffer(invalid_payload)
    payload_info = probe._vector_info(6, 1)
    assert payload_info is not None
    payload_start, _ = payload_info
    invalid_payload[payload_start + 4 : payload_start + 8] = b"NOPE"
    invalid_view = envelope.EnvelopeView.from_buffer(invalid_payload)
    invalid_dynamic = invalid_view.payload
    assert invalid_dynamic is not None
    with pytest.raises(InvalidBufferError, match="file identifier"):
        _ = invalid_dynamic.value
    with pytest.raises(InvalidBufferError, match="file identifier"):
        invalid_view.to_model()
    with pytest.raises(InvalidBufferError, match="file identifier"):
        envelope.Envelope.from_flatbuffer(invalid_payload)


def test_dynamic_view_rejects_disallowed_wire_tag(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, envelope = generated_modules
    buffer = _model(payload, envelope).to_flatbuffer()
    disallowed_tag = bytearray(buffer)
    probe = envelope.EnvelopeView.from_buffer(disallowed_tag)
    type_position = probe._field_position(4, 4)
    assert type_position is not None
    type_start = type_position + struct.unpack_from(
        "<I",
        disallowed_tag,
        type_position,
    )[0]
    assert struct.unpack_from("<I", disallowed_tag, type_start)[0] == len(METRIC_TAG)
    disallowed_tag[type_start + 4 : type_start + 4 + len(METRIC_TAG)] = (
        b"Outside.Dynamic.Metric"
    )
    with pytest.raises(ValueError, match="outside"):
        _ = envelope.EnvelopeView.from_buffer(disallowed_tag).payload


def test_unknown_allowed_dynamic_value_is_preserved(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, envelope = generated_modules
    opaque = envelope.EnvelopePayload.opaque(
        "Example.Dynamic.Future",
        b"unknown payload",
    )
    model = envelope.Envelope(payload=opaque, note="forward")

    builtins = msgspec.to_builtins(model, enc_hook=enc_hook)
    assert builtins["payload"] == {
        "__msgspec_flatbuffers_type__": "Example.Dynamic.Future",
        "__msgspec_flatbuffers_data__": "dW5rbm93biBwYXlsb2Fk",
    }
    assert msgspec.convert(
        builtins,
        type=envelope.Envelope,
        dec_hook=dec_hook,
    ) == model
    assert msgspec.json.decode(
        msgspec.json.encode(model, enc_hook=enc_hook),
        type=envelope.Envelope,
        dec_hook=dec_hook,
    ) == model

    unexpected = copy.deepcopy(builtins)
    unexpected["payload"]["extra"] = True
    with pytest.raises(msgspec.ValidationError, match="unexpected fields"):
        msgspec.convert(
            unexpected,
            type=envelope.Envelope,
            dec_hook=dec_hook,
        )

    view = envelope.EnvelopeView.from_buffer(model.to_flatbuffer())
    dynamic = view.payload
    assert dynamic is not None
    assert dynamic.tag == "Example.Dynamic.Future"
    assert not dynamic.is_known
    assert dynamic.value is None
    assert bytes(dynamic.data) == b"unknown payload"
    assert view.to_model() == model
    assert envelope.Envelope.from_flatbuffer(model.to_flatbuffer()) == model


def test_dynamic_payload_supports_scalar_and_vector_unions(tmp_path: Path) -> None:
    extension_source = tmp_path / "union_extension.fbs"
    extension_source.write_text(
        " ".join(
            (
                'attribute "dynamic_extension";',
                "namespace Perf.Dynamic;",
                "table Cat { name:string; }",
                "table Dog { name:string; }",
                "union Pet { Cat, Dog }",
                "table UnionExtension (dynamic_extension) {",
                "favorite:Pet; residents:[Pet]; }",
                "root_type UnionExtension;",
                'file_identifier "DUNN";',
            )
        ),
        encoding="utf-8",
    )
    envelope_source = tmp_path / "dynamic_envelope.fbs"
    envelope_source.write_text(
        " ".join(
            (
                'attribute "dynamic_flatbuffer";',
                'attribute "dynamic_allow";',
                "namespace Perf.Dynamic;",
                "table Envelope { payload_type:string; payload:[ubyte] (",
                'dynamic_flatbuffer: "payload_type",',
                'dynamic_allow: "Perf.Dynamic.*"); }',
                "root_type Envelope;",
                'file_identifier "DENV";',
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generated"

    generate(extension_source, output, project_root=tmp_path)
    generate(envelope_source, output, project_root=tmp_path)

    _clear_generated_modules("perf")
    with _temporary_sys_path(output):
        extension = importlib.import_module("perf.dynamic.union_extension")
        envelope = importlib.import_module("perf.dynamic.envelope")
        payload = extension.UnionExtension(
            favorite=extension.Cat(name="Miso"),
            residents=[extension.Dog(name="Tess"), extension.Cat(name="Luna")],
        )
        model = envelope.Envelope(payload=envelope.EnvelopePayload(payload))

        view = envelope.EnvelopeView.from_buffer(model.to_flatbuffer())
        dynamic = view.payload
        assert dynamic is not None
        nested = dynamic.value
        assert isinstance(nested, extension.UnionExtensionView)
        assert nested.favorite.name == "Miso"
        assert [resident.name for resident in nested.residents] == ["Tess", "Luna"]
        assert view.to_model() == model
        assert envelope.Envelope.from_flatbuffer(model.to_flatbuffer()) == model
    _clear_generated_modules("perf")


@pytest.mark.parametrize(
    ("vtable_entry_offset", "message"),
    [
        (4, "without a type tag"),
        (6, "type tag has no data"),
    ],
)
def test_dynamic_field_rejects_inconsistent_wire_values(
    generated_modules: tuple[ModuleType, ModuleType],
    vtable_entry_offset: int,
    message: str,
) -> None:
    payload, envelope = generated_modules
    buffer = bytearray(_model(payload, envelope).to_flatbuffer())
    probe = envelope.EnvelopeView.from_buffer(buffer)
    struct.pack_into("<H", buffer, probe._vtable_offset + vtable_entry_offset, 0)
    inconsistent = envelope.EnvelopeView.from_buffer(buffer)
    with pytest.raises(InvalidBufferError, match=message):
        _ = inconsistent.payload


def test_dynamic_registry_supports_concurrent_registration() -> None:
    class Model(msgspec.Struct):
        value: int

        def to_flatbuffer(self) -> bytes:
            return b"buffer"

    class View(TableView):
        def to_model(self) -> Model:
            return Model(1)

    class OtherView(View):
        pass

    tag = "Registry.Model"
    registry = type(dynamic_types)()
    assert registry.lookup_tag(tag) is None
    assert registry.lookup_model(Model) is None

    with ThreadPoolExecutor(max_workers=8) as executor:
        misses = list(executor.map(registry.lookup_tag, [tag] * 32))
    assert misses == [None] * 32

    entry = registry.register(tag, Model, View)
    assert registry.register(tag, Model, View) is entry
    assert registry.lookup_tag(tag) is entry
    assert registry.lookup_model(Model) is entry

    racing = type(dynamic_types)()
    barrier = Barrier(9)

    def lookup_tag() -> object:
        barrier.wait()
        return racing.lookup_tag(tag)

    def register_type() -> object:
        barrier.wait()
        return racing.register(tag, Model, View)

    with ThreadPoolExecutor(max_workers=9) as executor:
        readers = [executor.submit(lookup_tag) for _ in range(8)]
        writer = executor.submit(register_type)
    registered = writer.result()
    observed = [reader.result() for reader in readers]
    assert all(result in (None, registered) for result in observed)
    assert racing.lookup_tag(tag) is registered

    with pytest.raises(ValueError, match="conflicts"):
        registry.register(tag, Model, OtherView)


def test_dynamic_registry_invalidates_cached_subclass_resolution() -> None:
    class Model(msgspec.Struct):
        value: int

        def to_flatbuffer(self) -> bytes:
            return b"buffer"

    class Child(Model):
        pass

    class View(TableView):
        def to_model(self) -> Model:
            return Model(1)

    registry = type(dynamic_types)()
    assert registry.lookup_model(Child) is None

    entry = registry.register("Registry.Model", Model, View)

    assert registry.lookup_model(Child) is entry


def test_lazy_dynamic_module_imports_outside_registry_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model(msgspec.Struct):
        def to_flatbuffer(self) -> bytes:
            return b"buffer"

    class View(TableView):
        def to_model(self) -> Model:
            return Model()

    registry = type(dynamic_types)()
    assert registry.lookup_tag("Lazy.Model") is None
    registry.register_module("Lazy.Model", "trusted.lazy_model")
    imports: list[str] = []

    def load(module: str) -> object:
        imports.append(module)
        registry.register("Lazy.Model", Model, View)
        return object()

    monkeypatch.setattr(dynamic_module, "import_module", load)
    entry = registry.lookup_tag("Lazy.Model")
    assert entry is not None
    assert entry.model_type is Model
    assert imports == ["trusted.lazy_model"]


@pytest.mark.parametrize(
    ("declaration", "attributes", "message"),
    [
        (
            "other_type:string;",
            'dynamic_flatbuffer: "type_name", dynamic_allow: "Allowed.*"',
            "references missing type field",
        ),
        (
            "type_name:uint;",
            'dynamic_flatbuffer: "type_name", dynamic_allow: "Allowed.*"',
            "must be a string",
        ),
        (
            "type_name:string;",
            'dynamic_flatbuffer: "type_name"',
            "requires dynamic_allow",
        ),
        (
            "type_name:string;",
            'dynamic_flatbuffer: "type_name", dynamic_allow: "Allowed*"',
            "invalid dynamic_allow",
        ),
    ],
)
@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_invalid_dynamic_field_pairs_are_rejected(
    tmp_path: Path,
    declaration: str,
    attributes: str,
    message: str,
) -> None:
    source = tmp_path / "invalid_dynamic.fbs"
    source.write_text(
        " ".join(
            (
                'attribute "dynamic_flatbuffer";',
                'attribute "dynamic_allow";',
                "namespace Invalid;",
                f"table Root {{ {declaration} data:[ubyte] ({attributes}); }}",
                "root_type Root;",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(GenerationError, match=message):
        generate(source, tmp_path / "generated", project_root=tmp_path)


def test_generated_extension_root_registers_on_import(
    generated_modules: tuple[ModuleType, ModuleType],
) -> None:
    payload, _ = generated_modules
    entry = dynamic_types.lookup_tag(METRIC_TAG)
    assert entry is not None
    assert entry.model_type is payload.Metric
    assert entry.view_type is payload.MetricView


@pytest.mark.skipif(not HAS_FLATC, reason="flatc is not installed")
def test_dynamic_wrapper_and_table_names_are_scoped_by_module(tmp_path: Path) -> None:
    source = tmp_path / "dynamic_name_collision.fbs"
    source.write_text(
        " ".join(
            (
                'attribute "dynamic_flatbuffer";',
                'attribute "dynamic_allow";',
                "namespace Collision;",
                "table EnvelopePayload { value:int; }",
                "table Envelope { type:string; payload:[ubyte] (",
                'dynamic_flatbuffer: "type", dynamic_allow: "Collision.*"); }',
                "root_type Envelope;",
            )
        ),
        encoding="utf-8",
    )

    output = tmp_path / "generated"
    generate(source, output, project_root=tmp_path)

    _clear_generated_modules("collision")
    with _temporary_sys_path(output):
        envelope = importlib.import_module("collision.envelope")
        payload = importlib.import_module("collision.envelope_payload")

        assert envelope.EnvelopePayload is not payload.EnvelopePayload
        assert envelope.Envelope.__annotations__["payload"] == (
            "EnvelopePayload | None"
        )
    _clear_generated_modules("collision")

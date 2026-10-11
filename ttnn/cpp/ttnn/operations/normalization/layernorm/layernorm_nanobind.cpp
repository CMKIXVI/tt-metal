// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_nanobind.hpp"

#include <optional>
#include <new>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "layernorm.hpp"
#include "device/layernorm_device_operation.hpp"
#include "device/layernorm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "device/layernorm_types.hpp"
#include "device/layernorm_common.hpp"

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::operations::normalization::detail {

struct LayerNormProgramConfigPlaceholder {};

void bind_normalization_layernorm_types(nb::module_& mod) {
    export_enum<ttnn::prim::LayerNormType>(mod, "LayerNormType");
    export_enum<ttnn::prim::DistributedLayerNormStage>(mod, "DistributedLayerNormStage");
}

void bind_normalization_layernorm_program_config(nb::module_& mod) {
    nb::class_<LayerNormProgramConfigPlaceholder>(mod, "LayerNormProgramConfig");

    nb::class_<ttnn::prim::LayerNormDefaultProgramConfig>(mod, "LayerNormDefaultProgramConfig")
        .def(
            "__init__",
            [](prim::LayerNormDefaultProgramConfig* config, bool legacy_reduction, bool use_welford) {
                new (config) prim::LayerNormDefaultProgramConfig{
                    .legacy_reduction = legacy_reduction, .use_welford = use_welford};
            },
            nb::kw_only(),
            nb::arg("legacy_reduction").noconvert() = false,
            nb::arg("use_welford").noconvert() = false)
        .def_rw("legacy_reduction", &prim::LayerNormDefaultProgramConfig::legacy_reduction)
        .def_rw("use_welford", &prim::LayerNormDefaultProgramConfig::use_welford)
        .def("__repr__", [](const ttnn::prim::LayerNormDefaultProgramConfig& config) {
            return fmt::format("{}", config);
        });

    nb::class_<ttnn::prim::LayerNormShardedMultiCoreProgramConfig>(mod, "LayerNormShardedMultiCoreProgramConfig")
        .def(
            "__init__",
            [](prim::LayerNormShardedMultiCoreProgramConfig* config,
               CoreCoord compute_with_storage_grid_size,
               std::size_t subblock_w,
               std::size_t block_h,
               std::size_t block_w,
               bool inplace,
               bool legacy_reduction,
               bool use_welford) {
                new (config) prim::LayerNormShardedMultiCoreProgramConfig{
                    .compute_with_storage_grid_size = compute_with_storage_grid_size,
                    .subblock_w = subblock_w,
                    .block_h = block_h,
                    .block_w = block_w,
                    .inplace = inplace,
                    .legacy_reduction = legacy_reduction,
                    .use_welford = use_welford};
            },
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("subblock_w").noconvert(),
            nb::arg("block_h").noconvert(),
            nb::arg("block_w").noconvert(),
            nb::arg("inplace").noconvert(),
            nb::arg("legacy_reduction").noconvert() = false,
            nb::arg("use_welford").noconvert() = false)
        .def_rw(
            "compute_with_storage_grid_size",
            &prim::LayerNormShardedMultiCoreProgramConfig::compute_with_storage_grid_size)
        .def_rw("subblock_w", &prim::LayerNormShardedMultiCoreProgramConfig::subblock_w)
        .def_rw("block_h", &prim::LayerNormShardedMultiCoreProgramConfig::block_h)
        .def_rw("block_w", &prim::LayerNormShardedMultiCoreProgramConfig::block_w)
        .def_rw("inplace", &prim::LayerNormShardedMultiCoreProgramConfig::inplace)
        .def_rw("legacy_reduction", &prim::LayerNormShardedMultiCoreProgramConfig::legacy_reduction)
        .def_rw("use_welford", &prim::LayerNormShardedMultiCoreProgramConfig::use_welford)
        .def("__repr__", [](const ttnn::prim::LayerNormShardedMultiCoreProgramConfig& config) {
            return fmt::format("{}", config);
        });
}

void bind_normalization_layernorm_operation(nb::module_& mod) {
    mod.def(
        "layernorm_default_compute_config",
        &ttnn::layernorm_default_compute_config,
        nb::arg("arch"),
        R"doc(
        Returns the default compute kernel config for layernorm.
        )doc");

    mod.def(
        "create_layernorm_program_config",
        &ttnn::prim::create_layernorm_program_config,
        nb::arg("shard_spec") = nb::none(),
        nb::arg("tile_height") = 32,
        nb::arg("tile_width") = 32,
        R"doc(
        Creates a program config from shard spec.
        )doc");

    const auto* doc = R"doc(
        Computes layer norm over :attr:`input_tensor`.
        )doc";

    ttnn::bind_function<"layer_norm">(
        mod,
        doc,
        nb::overload_cast<
            const ttnn::Tensor&,
            float,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<const ttnn::Tensor>&,
            const std::optional<ttnn::MemoryConfig>&,
            const std::optional<const ttnn::prim::LayerNormProgramConfig>&,
            std::optional<const ttnn::DeviceComputeKernelConfig>,
            const std::optional<const ttnn::Tensor>&>(&ttnn::layer_norm),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("recip_tensor") = nb::none());
}

void bind_normalization_layernorm_params_and_inputs(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormParams>(mod, "LayerNormParams")
        .def(nb::init<>())
        .def_rw("norm_type", &ttnn::prim::LayerNormParams::norm_type)
        .def_rw("distributed_norm_stage", &ttnn::prim::LayerNormParams::distributed_norm_stage)
        .def_rw("eps", &ttnn::prim::LayerNormParams::eps)
        .def_rw("output_mem_config", &ttnn::prim::LayerNormParams::output_mem_config)
        .def_rw("program_config", &ttnn::prim::LayerNormParams::program_config)
        .def_rw("compute_kernel_config", &ttnn::prim::LayerNormParams::compute_kernel_config)
        .def_rw("dtype", &ttnn::prim::LayerNormParams::dtype);

    nb::class_<ttnn::prim::LayerNormInputs>(mod, "LayerNormInputs")
        .def(
            "__init__",
            [](ttnn::prim::LayerNormInputs* t, const ttnn::Tensor& input) {
                new (t) ttnn::prim::LayerNormInputs{
                    input, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
            },
            nb::arg("input"))
        .def_rw("input", &ttnn::prim::LayerNormInputs::input)
        .def_rw("residual_input_tensor", &ttnn::prim::LayerNormInputs::residual_input_tensor)
        .def_rw("weight", &ttnn::prim::LayerNormInputs::weight)
        .def_rw("bias", &ttnn::prim::LayerNormInputs::bias)
        .def_rw("stats", &ttnn::prim::LayerNormInputs::stats)
        .def_rw("recip_tensor", &ttnn::prim::LayerNormInputs::recip_tensor);
}

void bind_normalization_layernorm_device_operation(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormDeviceOperation>(mod, "LayerNormDeviceOperation")
        .def_static(
            "compute_program_hash",
            [](const ttnn::prim::LayerNormParams& attrs, const ttnn::prim::LayerNormInputs& tensors) {
                return ttnn::device_operation::detail::compute_program_hash<ttnn::prim::LayerNormDeviceOperation>(attrs, tensors);
            },
            nb::arg("operation_attributes"), nb::arg("tensor_args"))
        .def_static(
            "create_output_tensors",
            &ttnn::prim::LayerNormDeviceOperation::create_output_tensors,
            nb::arg("operation_attributes"), nb::arg("tensor_args"))
        .def_static(
            "compute_output_specs",
            &ttnn::prim::LayerNormDeviceOperation::compute_output_specs,
            nb::arg("operation_attributes"), nb::arg("tensor_args"))
        .def_static(
            "select_program_factory",
            &ttnn::prim::LayerNormDeviceOperation::select_program_factory,
            nb::arg("operation_attributes"), nb::arg("tensor_args"));
}

void bind_normalization_layernorm_program_factory(nb::module_& mod) {
    nb::class_<ttnn::prim::LayerNormMultiCoreProgramFactory>(mod, "LayerNormMultiCoreProgramFactory")
        .def_static(
            "default_core_range",
            &ttnn::prim::LayerNormMultiCoreProgramFactory::default_core_range,
            nb::arg("device"));

    nb::class_<ttnn::prim::LayerNormShardedProgramFactory>(mod, "LayerNormShardedProgramFactory");
}

void bind_normalization_layernorm(nb::module_& mod) {
    bind_normalization_layernorm_types(mod);
    bind_normalization_layernorm_program_config(mod);
    bind_normalization_layernorm_operation(mod);
    bind_normalization_layernorm_params_and_inputs(mod);
    bind_normalization_layernorm_device_operation(mod);
    bind_normalization_layernorm_program_factory(mod);
}

}  // namespace ttnn::operations::normalization::detail

// NOLINTEND(bugprone-unused-raii)

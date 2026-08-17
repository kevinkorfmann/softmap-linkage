"""Command-line interface for the SoftMap MVP."""

from __future__ import annotations

import argparse
import json

from .api import F2LinkageData, fit_f2, read_vcf
from .core import (
    fit_hierarchical_softmap,
    fit_scalable_likelihood_mds_ensemble,
    fit_softmap,
)
from .io import (
    read_probability_tsv,
    write_f2_result_tsv,
    write_hierarchical_result_tsv,
    write_likelihood_mds_result_tsv,
    write_result_tsv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softmap")
    parser.add_argument(
        "input", help="VCF/VCF.GZ/BCF or marker-by-offspring probability TSV"
    )
    parser.add_argument("output", help="output marker map TSV")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chromosome", help="VCF chromosome/contig to fit")
    parser.add_argument(
        "--cross-design",
        choices=("auto", "backcross", "ril", "doubled_haploid", "f2"),
        default="auto",
        help="cross design for VCF input",
    )
    parser.add_argument(
        "--physical-scaffold",
        action="store_true",
        help="use VCF physical positions to curate the final F2 marker order",
    )
    parser.add_argument(
        "--parents",
        nargs=2,
        metavar=("STATE0", "STATE1"),
        help="VCF parent samples; STATE0 is recurrent parent for a backcross",
    )
    parser.add_argument(
        "--likelihood-mds",
        action="store_true",
        help=(
            "use robust likelihood-MDS stability bands with automatic scalable "
            "likelihood binning above 500 markers"
        ),
    )
    parser.add_argument("--stability-mass", type=float, default=0.90)
    parser.add_argument(
        "--posterior-refinement-weight",
        type=float,
        default=0.75,
        help="blend weight for each interval-aware HMM refinement pass",
    )
    parser.add_argument("--maximum-posterior-refinement-passes", type=int, default=2)
    parser.add_argument(
        "--second-refinement-uncertain-pair-threshold",
        type=float,
        default=0.03,
    )
    parser.add_argument("--stability-rank-padding", type=int, default=1)
    parser.add_argument(
        "--minimum-stability-comparable-pair-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument("--smacof-iterations", type=int, default=500)
    parser.add_argument(
        "--no-posterior-calibration",
        action="store_true",
        help="disable the auditable residual-triggered likelihood-softening pass",
    )
    parser.add_argument("--bin-threshold", type=float, default=0.01)
    parser.add_argument(
        "--auto-bin",
        action="store_true",
        help="select a coarse threshold from pattern collapse and offspring information",
    )
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument(
        "--ordering-ensemble",
        action="store_true",
        help="select spectral, MST, or optimal-leaf order by multipoint likelihood",
    )
    parser.add_argument(
        "--hmm-lod",
        type=float,
        help="enable multipoint HMM framework densification at this log10 gap",
    )
    fine_binning = parser.add_mutually_exclusive_group()
    fine_binning.add_argument("--fine-bin-threshold", type=float)
    fine_binning.add_argument(
        "--coarse-only",
        action="store_true",
        help=(
            "disable the automatic finer HMM candidate layer used with --auto-bin "
            "when marker density is extreme"
        ),
    )
    parser.add_argument("--scaffold-confidence", type=float, default=0.85)
    parser.add_argument(
        "--prune-scaffold-lod",
        type=float,
        help="remove bootstrap anchors lacking this leave-one-out HMM LOD support",
    )
    parser.add_argument(
        "--post-scaffold-lod",
        type=float,
        help="reaudit bootstrap anchors after supported HMM insertions add context",
    )
    parser.add_argument(
        "--hmm-bootstrap",
        type=int,
        help="require HMM insertion positions to recur across this many bootstraps",
    )
    parser.add_argument("--hmm-position-support", type=float, default=0.8)
    parser.add_argument("--max-hmm-additions", type=int)
    insertion_mode = parser.add_mutually_exclusive_group()
    insertion_mode.add_argument("--greedy-hmm-pass", action="store_true")
    insertion_mode.add_argument(
        "--support-priority-pass",
        action="store_true",
        help="prioritize HMM candidates by positional likelihood support",
    )
    parser.add_argument(
        "--global-scaffold",
        action="store_true",
        help="require complete scaffold order support across bootstrap maps",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.hmm_lod is None and any(
        (
            args.fine_bin_threshold is not None,
            args.coarse_only,
            args.prune_scaffold_lod is not None,
            args.post_scaffold_lod is not None,
            args.hmm_bootstrap is not None,
            args.max_hmm_additions is not None,
            args.greedy_hmm_pass,
            args.support_priority_pass,
            args.global_scaffold,
        )
    ):
        parser.error("hierarchical/HMM controls require --hmm-lod")
    if args.coarse_only and not args.auto_bin:
        parser.error("--coarse-only is meaningful only with --auto-bin")
    if args.likelihood_mds and args.hmm_lod is not None:
        parser.error("--likelihood-mds and --hmm-lod are mutually exclusive")
    if args.no_posterior_calibration and not args.likelihood_mds:
        parser.error("--no-posterior-calibration requires --likelihood-mds")
    if args.physical_scaffold and args.cross_design != "f2":
        parser.error("--physical-scaffold requires --cross-design f2")
    if args.cross_design == "f2" and args.likelihood_mds:
        parser.error("F2 input uses its complete likelihood model automatically")
    lower_input = args.input.lower()
    data = None
    if lower_input.endswith((".vcf", ".vcf.gz", ".bcf")):
        data = read_vcf(
            args.input,
            chromosome=args.chromosome,
            parents=tuple(args.parents) if args.parents is not None else None,
            cross_design=args.cross_design,
        )
        names, probabilities = data.marker_names, data.probabilities
    else:
        if (
            args.chromosome is not None
            or args.parents is not None
            or args.cross_design != "auto"
        ):
            parser.error("VCF input options require a .vcf, .vcf.gz, or .bcf input")
        names, probabilities = read_probability_tsv(args.input)
    if isinstance(data, F2LinkageData):
        mapping = fit_f2(
            data,
            use_physical_scaffold=args.physical_scaffold,
            stability_mass=args.stability_mass,
            stability_rank_padding=args.stability_rank_padding,
            maximum_smacof_iterations=args.smacof_iterations,
        )
        write_f2_result_tsv(mapping.result, args.output)
        print(json.dumps(mapping.summary()))
        return
    bin_threshold = None if args.auto_bin else args.bin_threshold
    if args.likelihood_mds:
        result = fit_scalable_likelihood_mds_ensemble(
            probabilities,
            names,
            stability_mass=args.stability_mass,
            posterior_refinement_weight=args.posterior_refinement_weight,
            maximum_posterior_refinement_passes=(
                args.maximum_posterior_refinement_passes
            ),
            second_refinement_uncertain_pair_threshold=(
                args.second_refinement_uncertain_pair_threshold
            ),
            stability_rank_padding=args.stability_rank_padding,
            minimum_stability_comparable_pair_fraction=(
                args.minimum_stability_comparable_pair_fraction
            ),
            maximum_smacof_iterations=args.smacof_iterations,
            automatic_posterior_calibration=(not args.no_posterior_calibration),
        )
        write_likelihood_mds_result_tsv(result, args.output)
        summary = {
            "method": "SoftMap-LMDS-Ensemble",
            "status": result.status,
            "markers": len(names),
            "candidate_orders": int(result.candidate_orders.shape[0]),
            "likelihood_bins": result.likelihood_bin_count,
            "binning_method": result.binning_method,
            "bin_neighbor_count": result.bin_neighbor_count,
            "bin_neighbor_projection_dimensions": (
                result.bin_neighbor_projection_dimensions
            ),
            "ordering_method": result.ordering_method,
            "landmark_count": result.landmark_count,
            "landmark_neighbor_count": result.landmark_neighbor_count,
            "landmark_support_exponent": result.landmark_support_exponent,
            "large_scale_rescue_triggered": (result.large_scale_rescue_triggered),
            "low_certainty_stability_mass_cap_applied": (
                result.low_certainty_stability_mass_cap_applied
            ),
            "posterior_calibration_triggered": (result.posterior_calibration_triggered),
            "posterior_calibration_temperature": (
                result.posterior_calibration_temperature
            ),
            "uncalibrated_mean_genotype_certainty": (
                result.uncalibrated_mean_genotype_certainty
            ),
            "uncalibrated_distance_median_absolute_residual_morgan": (
                result.uncalibrated_distance_median_absolute_residual_morgan
            ),
            "selected_config": list(result.selected_config),
            "selection_method": result.selection_method,
            "weighted_objective_support_filter_applied": (
                result.weighted_objective_support_filter_applied
            ),
            "penalized_curve_effective_degrees_of_freedom": (
                result.penalized_curve_effective_degrees_of_freedom
            ),
            "posterior_refinement_weight": (result.posterior_refinement_weight),
            "posterior_refinement_passes_applied": (
                result.posterior_refinement_passes_applied
            ),
            "second_refinement_uncertain_pair_threshold": (
                result.second_refinement_uncertain_pair_threshold
            ),
            "stability_rank_padding": result.stability_rank_padding,
            "minimum_stability_comparable_pair_fraction": (
                result.minimum_stability_comparable_pair_fraction
            ),
            "unanimous_family_veto_triggered": (result.unanimous_family_veto_triggered),
            "stability_mass": result.stability_mass,
            "stability_comparable_pair_fraction": (
                result.stability_comparable_pair_fraction
            ),
            "distance_status": (
                result.genetic_distances.status
                if result.genetic_distances is not None
                else None
            ),
            "distance_method": (
                result.genetic_distances.method
                if result.genetic_distances is not None
                else None
            ),
            "map_length_cm": (
                result.genetic_distances.map_length_cm
                if result.genetic_distances is not None
                else None
            ),
            "distance_informative_pair_count": (
                result.genetic_distances.informative_pair_count
                if result.genetic_distances is not None
                else None
            ),
            "distance_segment_count": (
                result.genetic_distances.segment_count
                if result.genetic_distances is not None
                else None
            ),
            "distance_rank_span_weight_exponent": (
                result.genetic_distances.rank_span_weight_exponent
                if result.genetic_distances is not None
                else None
            ),
        }
    elif args.hmm_lod is None:
        result = fit_softmap(
            probabilities,
            names,
            confidence=args.confidence,
            bootstrap_replicates=args.bootstrap,
            bin_threshold=bin_threshold,
            neighbor_count=args.neighbors,
            ordering_ensemble=args.ordering_ensemble,
            random_seed=args.seed,
        )
        write_result_tsv(result, args.output)
        summary = {
            "method": "SoftMap",
            "status": (
                "ok" if result.framework.size >= 3 else "insufficient_order_information"
            ),
            "markers": len(names),
            "bins": int(result.bins.representatives.size),
            "bin_threshold": result.bins.threshold,
            "framework": int(result.framework.size),
            "effective_offspring_information": result.effective_offspring_information,
            "confidence": result.confidence,
        }
    else:
        auto_fine_bins = (
            args.auto_bin and args.fine_bin_threshold is None and not args.coarse_only
        )
        result = fit_hierarchical_softmap(
            probabilities,
            names,
            interval_confidence=args.confidence,
            scaffold_confidence=args.scaffold_confidence,
            bootstrap_replicates=args.bootstrap,
            coarse_bin_threshold=bin_threshold,
            fine_bin_threshold=args.fine_bin_threshold,
            auto_fine_bins=auto_fine_bins,
            neighbor_count=args.neighbors,
            ordering_ensemble=args.ordering_ensemble,
            min_log10_gap=args.hmm_lod,
            scaffold_prune_log10_gap=args.prune_scaffold_lod,
            post_scaffold_log10_gap=args.post_scaffold_lod,
            hmm_bootstrap_replicates=args.hmm_bootstrap,
            hmm_position_support=args.hmm_position_support,
            max_additions=args.max_hmm_additions,
            greedy_pass=args.greedy_hmm_pass,
            support_priority_pass=args.support_priority_pass,
            global_scaffold=args.global_scaffold,
            random_seed=args.seed,
        )
        write_hierarchical_result_tsv(result, args.output)
        summary = {
            "method": "SoftMap-HMM",
            "status": result.status,
            "markers": len(names),
            "coarse_bins": int(result.support.bins.representatives.size),
            "coarse_bin_threshold": result.support.bins.threshold,
            "candidate_bins": int(result.bins.representatives.size),
            "fine_bin_threshold": result.fine_bin_threshold,
            "scaffold": int(result.scaffold.size),
            "framework": int(result.framework.size),
            "min_log10_gap": result.min_log10_gap,
            "scaffold_prune_log10_gap": result.scaffold_prune_log10_gap,
            "post_scaffold_log10_gap": result.post_scaffold_log10_gap,
            "hmm_bootstrap_replicates": result.hmm_bootstrap_replicates,
            "hmm_position_support": result.hmm_position_support,
        }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

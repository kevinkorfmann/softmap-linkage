import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import softmap

VCF = """##fileformat=VCFv4.3
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred genotype likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tp0\tp1\to1\to2\to3
chr1\t10\tm1\tA\tG\t60\tPASS\t.\tGT:PL\t0/0:0,90,120\t1/1:120,90,0\t0/0:0,20,80\t0/1:20,0,80\t./.:.,.,.
chr1\t20\t.\tC\tT\t60\tPASS\t.\tGT\t1/1\t0/0\t1/1\t0/1\t0/1
"""


class VCFInputTests(unittest.TestCase):
    def test_parent_oriented_f2_retains_all_three_genotypes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "family.vcf"
            path.write_text(VCF)
            data = softmap.read_vcf(
                path,
                chromosome="chr1",
                parents=("p0", "p1"),
                cross_design="f2",
            )
        self.assertIsInstance(data, softmap.F2LinkageData)
        self.assertEqual(data.probabilities.shape, (3, 2, 3))
        self.assertEqual(int(np.argmax(data.probabilities[0, 0])), 0)
        self.assertEqual(int(np.argmax(data.probabilities[1, 0])), 1)
        np.testing.assert_allclose(data.probabilities[2, 0], [0.25, 0.5, 0.25])
        # The parents are REF/ALT-reversed at marker two; parental-state
        # orientation must remain consistent across the chromosome.
        self.assertEqual(int(np.argmax(data.probabilities[0, 1])), 0)
        self.assertEqual(int(np.argmax(data.probabilities[1, 1])), 1)

    def test_parent_oriented_backcross_vcf(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "family.vcf"
            path.write_text(VCF)
            data = softmap.read_vcf(
                path,
                chromosome="chr1",
                parents=("p0", "p1"),
                cross_design="backcross",
            )
        self.assertEqual(data.marker_names, ("m1", "chr1:20"))
        self.assertEqual(data.probabilities.shape, (3, 2))
        self.assertLess(data.probabilities[0, 0], 0.02)
        self.assertGreater(data.probabilities[1, 0], 0.98)
        self.assertEqual(data.probabilities[2, 0], 0.5)
        np.testing.assert_allclose(data.physical_positions, [10.0, 20.0])

    def test_gzipped_vcf(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "family.vcf"
            source.write_text(VCF)
            data = softmap.read_vcf(
                source,
                parents=("p0", "p1"),
                cross_design="backcross",
            )
            path = Path(directory) / "family.vcf.gz"
            import pysam

            pysam.tabix_compress(str(source), str(path), force=True)
            compressed = softmap.read_vcf(
                path,
                parents=("p0", "p1"),
                cross_design="backcross",
            )
        np.testing.assert_allclose(compressed.probabilities, data.probabilities)

    def test_parents_require_explicit_cross_design(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "family.vcf"
            path.write_text(VCF)
            with self.assertRaisesRegex(ValueError, "set cross_design"):
                softmap.read_vcf(path, parents=("p0", "p1"))

    def test_bcf_input(self):
        import pysam

        with TemporaryDirectory() as directory:
            vcf_path = Path(directory) / "family.vcf"
            vcf_path.write_text(VCF)
            bcf_path = Path(directory) / "family.bcf"
            with (
                pysam.VariantFile(str(vcf_path)) as source,
                pysam.VariantFile(str(bcf_path), "wb", header=source.header) as output,
            ):
                for record in source:
                    output.write(record)
            data = softmap.read_vcf(
                bcf_path,
                parents=("p0", "p1"),
                cross_design="backcross",
            )
        self.assertEqual(data.marker_names, ("m1", "chr1:20"))

    def test_haploid_doubled_haploid_calls(self):
        text = """##fileformat=VCFv4.3
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tp0\tp1\to1\to2
chr1\t10\tm1\tA\tG\t60\tPASS\t.\tGT\t0\t1\t0\t1
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dh.vcf"
            path.write_text(text)
            data = softmap.read_vcf(
                path,
                parents=("p0", "p1"),
                cross_design="doubled_haploid",
            )
        np.testing.assert_allclose(data.probabilities[:, 0], [0.01, 0.99])


if __name__ == "__main__":
    unittest.main()

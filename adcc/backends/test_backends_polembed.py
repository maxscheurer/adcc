#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import unittest
import itertools
import adcc
import adcc.backends

from adcc.adc_pp.matrix import AdcBlock
from adcc import OneParticleOperator, AmplitudeVector

from numpy.testing import assert_allclose

import pytest

from ..misc import expand_test_templates
from .testing import cached_backend_hf
from ..testdata.cache import qchem_data, tmole_data
from ..testdata.static_data import pe_potentials

try:
    import cppe  # noqa: F401

    has_cppe = True
except ImportError:
    has_cppe = False


backends = [b for b in adcc.backends.available()
            if b not in ["molsturm", "veloxchem"]]
basissets = ["sto3g", "ccpvdz"]
methods = ["adc1", "adc2", "adc3"]


@pytest.mark.skipif(not has_cppe, reason="CPPE not found")
@pytest.mark.skipif(len(backends) == 0, reason="No backend found.")
@expand_test_templates(list(itertools.product(basissets, methods, backends)))
class TestPolarizableEmbedding(unittest.TestCase):
    def template_pe_perturbative_formaldehyde(self, basis, method, backend):
        basename = f"formaldehyde_{basis}_pe_{method}"
        qc_result = qchem_data[basename]
        pe_options = {"potfile": pe_potentials["fa_6w"]}
        scfres = cached_backend_hf(backend, "formaldehyde", basis,
                                   pe_options=pe_options)
        state = adcc.run_adc(scfres, method=method,
                             n_singlets=5, conv_tol=1e-10)
        assert_allclose(
            qc_result["excitation_energy"],
            state.excitation_energy_uncorrected,
            atol=1e-5
        )
        assert_allclose(
            + qc_result["excitation_energy"]
            + qc_result["pe_ptss_correction"]
            + qc_result["pe_ptlr_correction"],
            state.excitation_energy,
            atol=1e-5
        )
        assert_allclose(
            qc_result["pe_ptss_correction"],
            state.pe_ptss_correction,
            atol=1e-5
        )
        assert_allclose(
            qc_result["pe_ptlr_correction"],
            state.pe_ptlr_correction,
            atol=1e-5
        )

    def template_pe_coupling_formaldehyde(self, basis, method, backend):
        if method != "adc2":
            pytest.skip("")
        basename = f"formaldehyde_{basis}_pe_{method}"
        tm_result = tmole_data[basename]
        pe_options = {"potfile": pe_potentials["fa_6w"]}
        scfres = cached_backend_hf(backend, "formaldehyde", basis,
                                   pe_options=pe_options)
        assert_allclose(scfres.energy_scf, tm_result["energy_scf"], atol=1e-8)

        # additional matrix apply for PE
        def block_ph_ph_0_pe(hf, mp, intermediates):
            op = hf.operators

            def apply(ampl):
                tdm = OneParticleOperator(mp, is_symmetric=False)
                tdm.vo = ampl.ph.transpose()
                vpe = op.density_dependent_operators["pe_induction_elec"](tdm)
                return AmplitudeVector(ph=vpe.ov)
            return AdcBlock(apply, 0)

        matrix = adcc.AdcMatrix(method, scfres, solvent_block=block_ph_ph_0_pe)
        assert_allclose(
            matrix.ground_state.energy(2),
            tm_result["energy_mp2"],
            atol=1e-8
        )
        state = adcc.run_adc(matrix, n_singlets=5, conv_tol=1e-7)
        assert_allclose(
            state.excitation_energy_uncorrected,
            tm_result["excitation_energy"],
            atol=1e-6
        )

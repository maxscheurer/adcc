from adcc import AdcMatrix
from .OneParticleOperator import OneParticleOperator


class PeAdcMatrix(AdcMatrix):
    def block_apply(self, block, in_vec):
        ret = super().block_apply(block, in_vec)
        if block == "ph_ph":
            # CIS-like contribution only adds to ph_ph block
            with self.timer.record("apply/pe_coupling"):
                op = self.reference_state.operators
                tdm = OneParticleOperator(self.ground_state, is_symmetric=False)
                tdm.vo = in_vec.ph.transpose()
                vpe = op.density_dependent_operators["pe_induction_elec"](tdm)
                ret.ph += vpe.ov
        return ret

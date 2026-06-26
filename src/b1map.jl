export roughness_penalty

"""
Eq 41 in regularized b1 mapping paper.
TODO: Assuming that kappa = 1. Need to allow kappa to not be 1.
"""
function roughness_penalty(z)
    # Offsets to compute neighboring pixel differences.
    nl_mls = ((1,0), (0, 1))
    N, M = size(z)
    penalty = 0.0
    for offset in nl_mls
        n_l = offset[1]
        m_l = offset[2]
        for n=1+abs(n_l):N-abs(n_l)
            for m=1+abs(m_l):M-abs(m_l)
                penalty += (2 * z[n, m] - z[n - n_l, m - m_l] - z[n + n_l, m+m_l])^2
            end
        end
    end

    return penalty
end
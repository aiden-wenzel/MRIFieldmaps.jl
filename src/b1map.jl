export roughness_penalty, regularizer

"""
Eq 41 in regularized b1 mapping paper.
TODO: Assuming that kappa = 1. Need to allow kappa to not be 1.
"""
function roughness_penalty(z)
    # Offsets to compute neighboring pixel differences.
    N, M = size(z)

    # TODO: Currently these offsets are only the left, right, top, bottom neighbors of each pixel.
    # Should the user be able to change the offsets they want to use for the roughness penalty?
    nl_mls = ((1,0), (0, 1)) 
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

"""
Eq. 6 in regularized b1 mapping paper.
"""
function regularizer(zks)
    K = size(zks, 1)
    regularized_cost = 0.0
    for k=1:K
        regularized_cost += roughness_penalty(zks[k])
    end
    return regularized_cost
end
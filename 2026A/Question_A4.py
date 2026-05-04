# starting with (A.2.8):
# b_k = dim(Z_k) - dim(B_k) = dim ker(d_k) - dim im(d_k+1)
# let D_k be the matrix of the boundary operator:
# d_k : C_k(K) -> C_k-1(K)
# then:
# dim im(d_k) = rank(D_k)
# so:
# dim B_k = dim im(d_k+1) = rank(D_k+1)
# using the rank-nullity theorem on d_k:
# dim C_k dim ker(d_k) + rank(D_k)
# and rearranging:
# dim ker(d_k) = dim C_k - rank(D_k)
# but dim C_k is just ```len(complex[k])```
# therefore:
# dim Z_k = dim ker(d_k) = dim C_k - rank(D_k)
# and finally substituting this into the Betti number formula:
# b_k = dim Z_k - dim B_k
# gives you:
# b_k = dim C_k - rank(d_k) - rank(d_k+1)

#! python

# --------- Modify the Following Based on Your Data Size ------------

#DATA_1 configuration
P_Nt = 4
P_tiling_factor = 8
P_Vsnk = 128
B2Nrows = 4
Nsrc = 44
s = 3  # Fixed
Nperms = 36
Nw2 = 288


# ----------- Do Not Modify The Rest --------------------------------

# Automatic Replacement
Lt = P_Nt
t = Lt
tiling_factor = P_tiling_factor
tile1 = tiling_factor
tile2 = tile1
Vsnk = P_Vsnk
x1 = Vsnk / tiling_factor
x2 = x1
r = B2Nrows
rp = r
m = Nsrc
nperm = Nperms
wnum = Nw2


# BB_BB_new_term_1_r1_b2("BB_BB_new_term_1_r1_b2", {t, tile1, tile2, x1, rp, x2, m, r, s, nperm, wnum} = 
Index_Size = t * tile1 * tile2 * x1 * rp * x2 * m * r * s * nperm * wnum

print "Index_Size = ", Index_Size, " = ", Index_Size/1000000000, " Billion"
print "Max_Signed_Int = 2 Billion"
print "Max_Signed_Int - Index_Size = 2147483647 - ", Index_Size, " = ", 2147483647 - Index_Size
print "Max_Signed_Int - Index_Size = 2.147483647 - ", Index_Size/1000000000, " Billion = ", (2147483647 - Index_Size)/1000000000, " Billion"


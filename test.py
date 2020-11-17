import islpy as isl

s.apply(sch1.reverse()).apply(k).intersect(r).is_empty()
True
k = isl.Map("{S0[i, j] -> S0[i' = 1 + i, j' = j]}") // dep
r = s.apply(sch1.reverse()) // result
s = isl.Set("[n]->{S0[i,j]:i=n}")
sch = isl.Map("{ S0[i, j] -> S0[i + j, j] }") schezdule

r = isl.Set("[n]->{S0[i,j]:i=n}").apply(isl.Map("{ S0[i, j] -> S0[i + j, j] }").reverse())

>>>r.apply(isl.Map("{S0[i, j] -> S0[i' = 1 + i, j' = j]}"))


s.apply(sch.reverse()).apply(dep).apply(sch).lex_ge_set(s).is_empty()



solve and rank before/after :



isl.Map.lex_lt(s.get_space()).intersect(isl.Map("{S0[i,j]->S0[i-1,j+10]}")).is_empty()



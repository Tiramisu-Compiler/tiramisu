##Name of the system
sys="gemm";
##Read the alphabets system of equations from .ab file
prog=ReadAlphabets(sys+".ab");

CheckProgram(prog);

##Print the AST to figure out the node id of the reduction. 
##Then use it to serialize the reduction.
#PrintAST(prog);

#Serializes the reduction
# Target is specified with the nodeID, which can be viewed with PrintAST(prog)
SerializeReduction(prog,"(0,0,0,0,1)","(i,j,k->i,j,k)");
Normalize(prog);

##The Alpha program after serialize the reduction 
#affine gemm {P,Q,R|P>=1 && Q>=1 && R>=1}
#input
#  double alpha {|};
#  double beta {|};
#  double A {i,j|P>=i+1 && j>=0 && Q>=j+1 && i>=0};
#  double B {i,j|Q>=i+1 && j>=0 && R>=j+1 && i>=0};
#  double Cin {i,j|P>=i+1 && j>=0 && R>=j+1 && i>=0};
#output
#  double Cout {i,j|P>=i+1 && j>=0 && R>=j+1 && i>=0};
#local
#  double _serCout {i,j,k|R>=j+1 && k>=0 && Q>=k+1 && i>=0 && P>=i+1 && j>=0};
#let
#  Cout[i,j] = ((alpha * _serCout[i,j,Q-1]) + (beta * Cin));
#  _serCout[i,j,k] = case
#    {|k>=1} : (_serCout[i,j,k-1] + (A[i,k] * B[k,j]));
#    {|k==0} : (A[i,k] * B[k,j]);
#   esac;
#.

###Space time map for local and output variables--------

##ikj
mapping="ikj";
setSpaceTimeMap(prog,sys,"Cout","(i,j->i,Q-1,j)");
setSpaceTimeMap(prog,sys,"_serCout","(i,j,k->i,k,j)");

##ijk
#mapping="ijk";
#setSpaceTimeMap(prog,sys,"Cout","(i,j->i,j,Q-1)");
#setSpaceTimeMap(prog,sys,"_serCout","(i,j,k->i,j,k)");
###-----------------------------------------------------

setStatementOrdering(prog, sys,"_serCout","Cout");

##Provide Memory Map for memory allocation for the variable "_serCout"
setMemoryMap(prog, sys, "_serCout", "Cout", "(i,j,k->i,j)");

###Parallel---------------------------------------------
##sequential
#type="sequential";

##parallel
type="openmp";
##if untiled and parallel
#setParallel(prog,sys,"","0");
###-----------------------------------------------------

###Tiling-----------------------------------------------
##with tiling
tiling="tiled";
options = createTiledCGOptionForScheduledC();
setDefaultDTilerConfiguration(prog, sys, type);
##separate full and partial tiles
setTiledCGOptionOptimize(options, 1);

##without tiling
#tiling="untiled";
#options = createCGOptionForScheduledC();
###-----------------------------------------------------

setCGOptionFlattenArrays(options, 1);

##source location
outDir="./src/"+sys+"/"+tiling+"-"+mapping+"/"+type;

##generate gemm code
generateScheduledCode(prog, sys, options, outDir);

##generate the wrapper class with main function
generateWrapper(prog, sys, options, outDir);

##generate Makefile
generateMakefile(prog, sys, options, outDir);

print(sys+" is Done.");
#==============================================================================

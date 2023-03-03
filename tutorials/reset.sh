for bra in tutorial_autoscheduler/*/*.py
do
	echo $bra
	sed -i -E "s/\/home\/nassim\/Desktop\/tiramisu_raw/\/data\/scratch\/tiramisu_git\/tiramisu_lanka\/tiramisu/gi" $bra
done
for bra in tutorial_autoscheduler/*/generator.cpp
do
	echo $bra
	sed -i -E "s/\/home\/nassim\/Desktop\/tiramisu_raw/\/data\/scratch\/tiramisu_git\/tiramisu_lanka\/tiramisu/gi" $bra
done
for bra in tutorial_autoscheduler/*.sh
do
	echo $bra
	sed -i -E "s/\/home\/nassim\/Desktop\/tiramisu_raw/\/data\/scratch\/tiramisu_git\/tiramisu_lanka\/tiramisu/gi" $bra
done

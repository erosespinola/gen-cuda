rm benchmark-qa194.csv

for pop_size in 2 4 8 16 32 64 128 256 512 1024
do
	echo $pop_size

	for step_size in 100 200 300 400 500 600 700 800 900 1000
	do
		echo -n $pop_size >> benchmark-qa194.csv
		echo -n "," >> benchmark-qa194.csv
		echo -n $step_size >> benchmark-qa194.csv
		echo -n "," >> benchmark-qa194.csv
		./gpu-ga -n $pop_size -s $step_size -i qa194.in -o results/qa194-gpu${pop_size}-${step_size}.txt >> benchmark-qa194.csv
		echo -n "," >> benchmark-qa194.csv
		./serial -n $pop_size -s $step_size -i qa194.in -o results/qa194-serial${pop_size}-${step_size}.txt >> benchmark-qa194.csv
		echo >> benchmark-qa194.csv
	done
done
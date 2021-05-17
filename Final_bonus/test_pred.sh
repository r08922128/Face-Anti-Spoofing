for epoch in {30..50}
do
	winpty python main.py --mode test --load $epoch
done
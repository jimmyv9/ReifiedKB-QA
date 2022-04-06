#!/bin/sh

#for ((a=0; a<95000; a+=5000));
for ((a=20000; a<30001; a+=5000));
do
		declare -i b=a+5000;
		outfile=./results/nhop1/to_train_$a.txt
		echo $outfile
		python read_metaQA.py ../MetaQA/kb.txt ../MetaQA/1-hop/vanilla/qa_train.txt word2vec-google-news-300 $outfile $a $b
done
#python read_metaQA.py ../MetaQA/kb.txt ../MetaQA/1-hop/vanilla/qa_train.txt word2vec-google-news-300 ./results/to_train_95000.txt 95000 -1

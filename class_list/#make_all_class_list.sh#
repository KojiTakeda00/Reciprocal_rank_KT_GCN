for train_date in 2015-08-28-09-50-22 2015-10-30-13-52-14 2015-11-10-10-32-52 2015-11-12-13-27-51 2015-11-13-10-28-08;do
    for test_date in 2015-08-28-09-50-22 2015-10-30-13-52-14 2015-11-10-10-32-52 2015-11-12-13-27-51 2015-11-13-10-28-08;do
	if [ $train_date == $test_date ];then
	    echo same
	else
	mkdir train_${train_date}_test_${test_date}
	#a=`cat ../full_size_img/by_origin_train-${train_date}_test${test_date}/class_list_train_${train_date}.txt | tail -n 1 `
	cp ../full_size_img/by_origin_train-${train_date}_test${test_date}/class_list_train_${train_date}.txt train_${train_date}_test_${test_date}
	cp ../full_size_img/by_origin_train-${train_date}_test${test_date}/class_list_test_${test_date}.txt train_${train_date}_test_${test_date}
	#echo ${a:20}
	fi
    done
done

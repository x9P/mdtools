#!/bin/sh

python3 ../generate_resp_points.py h2o.log
res=FAIL
diff -q esppoints_14{,.sav} && diff -q esppoints_16{,.sav} && diff -q esppoints_18{,.sav} && diff -q esppoints_20{,.sav} && res=PASS
echo $res

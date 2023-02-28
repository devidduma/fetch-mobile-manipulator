# generate csv
$ python ./tools.py --root-dir ./log/FetchReach-v3/ddpg
# generate figures
$ python ./plotter.py --root-dir ./log/FetchReach-v3/ddpg --shaded-std --legend-pattern "\\w+"
$ python ./plotter.py --root-dir "./log/FetchReach-v3" --shaded-std --legend-pattern "\\w+"
# generate numerical result (support multiple groups: `--root-dir ./` instead of single dir)
$ python ./analysis.py --root-dir ./log --norm

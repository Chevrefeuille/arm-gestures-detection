#!/usr/bin/gnuplot
#!/opt/local/bin/gnuplot


# wxt
#set terminal aqua
set terminal wxt size 480,360 enhanced font 'Verdana,10' persist


set autoscale                     

set grid ytics lt 0 lw 2 lc rgb "#bbbbbb"
set grid xtics lt 0 lw 2 lc rgb "#bbbbbb"

set border linewidth 1.5

# color definitions
set  linestyle 1 linecolor rgb '#0060ad' lt 1 lw 5 pt 7 ps 0.3   # --- blue
set  linestyle 2 linecolor rgb '#ad0009' lt 1 lw 5 pt 7 ps 0.3   # --- red


set ytics 1
set tics scale 0.75

set xrange [0:2]
set yrange [0:3]


set xlabel "t (s)"
set ylabel "Angle (°)"

plot '../data/gnuplot/no_gest_amdf.txt' using 1:2 with lines ls 1 title "amdf", \
     '' using 1:3  with lines ls 2 dashtype 4 title "fitted sine", \
     
set key right bottom

set terminal pdf
set output '../results/no_gest_amdf.pdf'
replot 
set terminal x11

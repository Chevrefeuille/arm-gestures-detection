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
set  linestyle 3 linecolor rgb '#008000' lt 1 lw 5 pt 7 ps 0.3   # --- green
set  linestyle 4 linecolor rgb '#ffa500' lt 1 lw 5 pt 7 ps 0.3   # --- orange

set ytics 0.5
set tics scale 0.75

set xrange [0:2.5]
set yrange [0:2.5]
#set xtics ( "-3" -3000, "-1.5" -1500, "0" 0, "1.5" 1500, "3" 3000)

set xlabel "v_g (m/s)"
set ylabel "p(v_g)"

plot '../data/pdfs/v_G_0.txt' with lines  ls 1  title "0", \
     '../data/pdfs/v_G_1.txt' with lines  ls 2  title "1", \
     '../data/pdfs/v_G_2.txt' with lines  ls 3  title "2", \
     '../data/pdfs/v_G_3.txt' with lines  ls 4  title "3", \
 
set key right top

set terminal pdf
set output '../results/vg_pdfs_intensity.pdf'
replot 
set terminal x11

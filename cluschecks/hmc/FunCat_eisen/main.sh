cd ~/CLUSDIR/
javac -d "bin" -cp ".:jars/commons-math-1.0.jar:jars/jgap.jar" clus/Clus.java
cd ~/PCT/cluschecks/hmc/FunCat_eisen
java -cp "$HOME/CLUSDIR/bin:$HOME/CLUSDIR/jars/commons-math-1.0.jar:$HOME/CLUSDIR/jars/jgap.jar" clus.Clus FunCat.s

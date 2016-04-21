package org.insightcentre.mono.aligners.nn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.AlignmentTrainer;
import org.insightcentre.mono.aligners.SentenceVectors;

/**
 *
 * @author John McCrae
 */
public class DFAlignerWeighting implements AlignmentTrainer<DFAlignerWeighting.DFWeightedAligner> {
    
    private final AlignmentTrainer alignerTrainer;

    public DFAlignerWeighting(AlignmentTrainer alignerTrainer) {
        this.alignerTrainer = alignerTrainer;
    }

    public static class DFWeightedAligner implements Aligner {
        private final Map<String, Double> weights;
        private final Aligner aligner;

        public DFWeightedAligner(Map<String, Double> weights, Aligner aligner) {
            this.weights = weights;
            this.aligner = aligner;
        }

        @Override
        public Alignment align(SentenceVectors x, SentenceVectors y) {
            final Alignment a = aligner.align(x, y);
            double[][] m = new double[x.words().length][y.words().length];
            for(int i = 0; i < x.words().length; i++) {
                for(int j = 0; j < y.words().length; j++) {
                    m[i][j] = (weights.containsKey(x.word(i)) ? 
                        weights.get(x.word(i)) : 1.0) *
                        (weights.containsKey(y.word(j)) ?
                        weights.get(y.word(j)) : 1.0) *
                        a.alignment(i, j);
                }
            }
            return new Alignment(x, y, m);
        }

        @Override
        public void save(File file) throws IOException {
            final File dfFile = new File(file.getAbsolutePath() + ".df");
            try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(dfFile))) {
                oos.writeObject(weights);
            }
            aligner.save(file);
        }

    }
    

    @Override
    public DFWeightedAligner train(List<AlignmentTrainer.TrainingPair> data) {
        final Map<String, Double> wts = new HashMap<>();
        int N = 0;
        for(TrainingPair tp : data) {
            for(String w : new HashSet<>(Arrays.asList(tp.x.words()))) {
                wts.put(w, wts.containsKey(w) ? wts.get(w) + 1 : 1);
            }
            for(String w : new HashSet<>(Arrays.asList(tp.y.words())))  {
                wts.put(w, wts.containsKey(w) ? wts.get(w) + 1 : 1);
            }
            N += 2;
        }

        for(Map.Entry<String, Double> wt : wts.entrySet()) {
            wt.setValue(1.0 - wt.getValue() / N);
        } 
        final Aligner aligner = alignerTrainer.train(data);
        return new DFWeightedAligner(wts, aligner);
    }

    @Override
    public DFWeightedAligner load(File file) throws IOException {
        final File dfFile = new File(file.getAbsolutePath() + ".df");
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(dfFile))) {
            final HashMap<String, Double> weights = (HashMap<String, Double>)ois.readObject();
            final Aligner aligner = alignerTrainer.load(file);
            return new DFWeightedAligner(weights, aligner);
        } catch (ClassNotFoundException ex) {
           throw new IOException(ex); 
        }
    }
    
}

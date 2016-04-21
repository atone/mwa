package org.insightcentre.mono.aligners.nn;

import edu.insight.unlp.nn.common.nlp.GloveVectors;
import edu.insight.unlp.nn.common.nlp.Word2Vector;
import java.io.File;
import java.io.IOException;
import static java.lang.Double.min;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.AlignmentTrainer;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.featextractors.StopWordRemoval;
import org.insightcentre.mono.aligners.util.IntPair;

/**
 * Loosely inspired by Sultan but much much faster 
 * 
 * @author John McCrae
 */
public class EasyMonolingualAligner implements Aligner {
	private static String embeddingPath = "models/glove.6B.50d.txt";
	
	private static Word2Vector w2vec = new GloveVectors(embeddingPath); 
    private static final double THRESHOLD = 0.90;
	
    private Set<String> morpho(String s2) {
        String s = s2.toLowerCase();
        if(s.matches(".*[b-df-hj-np-tv-z]ed")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3),  // batted
                                    s.substring(0, s.length() - 2),  // waited
                                    s.substring(0, s.length() - 1))); // loved
        }
        if(s.matches(".*[b-df-hj-np-tv-z]ing")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 4), // batting
                                    s.substring(0, s.length() - 3), // waiting
                                    s.substring(0, s.length() - 3) + "e")); // loving

        }
        if(s.matches(".*ied")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3) + "y")); // tried
        }
        if(s.matches(".*[aeiou]ed")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 2), //vetoed
                s.substring(0, s.length() - 1))); // peed
        }
        if(s.matches(".*ies")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3) + "y", // parties
                                    s.substring(0, s.length() - 1)));      // vies
        }
        if(s.matches(".*es")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 2), // watches
                                    s.substring(0, s.length() - 1))); // loves
        }
        if(s.matches(".*s")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() -1))); // cats
        }
        if(s.matches(".*[bcdfghjklmnpqrstvwxyz]er")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3), // fitter
                                    s.substring(0, s.length() - 2), // ?
                                    s.substring(0, s.length() - 1))); // cuter
        }
        if(s.matches(".*ier")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3) + "y", // happier
                                    s.substring(0, s.length() - 1))); // ?
        }
        if(s.matches(".*[aeiou]er")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 2), 
                                    s.substring(0, s.length() - 1)));
        }
        if(s.matches(".*[bcdfghjklmnpqrstvwxyz]est")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 4), // fittest
                                    s.substring(0, s.length() - 3), // ?
                                    s.substring(0, s.length() - 2))); // cutest
        }
        if(s.matches(".*iest")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 4) + "y", // happiest
                                    s.substring(0, s.length() - 2))); // ?
        }
        if(s.matches(".*[aeiou]est")) {
            return new HashSet<>(
                   Arrays.asList(s, s.substring(0, s.length() - 3), // 
                                    s.substring(0, s.length() - 2)));
        }
        return new HashSet<>(
               Arrays.asList(s));
    }

    public List<IntPair> align(String[] x, String[] y) {
        List<IntPair> result = new ArrayList<>();
        Set<String>[] ylemmas = new Set[y.length];
        for(int j = 0; j < y.length; j++) {
            ylemmas[j] = morpho(y[j]);
        }
        for(int i = 0; i < x.length; i++) {
            Set<String> variants = morpho(x[i]);
            int X = variants.size();
            int j = 0;
            for(int j2 = 0; j2 < y.length; j2++) {
                j = (i + j2) % y.length;
                variants.removeAll(ylemmas[j]);
                if(variants.size() != X)
                    break;
                j = y.length;
            }
            if(j < y.length) {
                result.add(new IntPair(i,j));
            } else if(!StopWordRemoval.stopwords.contains(x[i].toLowerCase())) {
                double[] xvec = w2vec.getWordVector(x[i]);
                
                double max = THRESHOLD;
                int j2 = -1;
                for(int j3 = 0; j3 < y.length; j3++) {
                    j = (i + j3) % y.length;
                //for(int j3 : new Near(i < y.length ? i : y.length - 1, y.length)) {
                    if(!StopWordRemoval.stopwords.contains(y[j].toLowerCase()) &&
                        x[i].matches("\\W+") == y[j].matches("\\W+")) {
                        double sim = WordSimAligner.cosine(xvec, w2vec.getWordVector(y[j]));
                        if(sim > max && sim != 1.0) {
                            max = sim;
                            j2 = j;
                        }
                    }
                }
                if(j2 != -1) {
                    result.add(new IntPair(i,j2));
                }
            }
        }
        return result;
    }
   @Override
    public Alignment align(SentenceVectors x, SentenceVectors y) {
        double[][] matrix = new double[x.size()][y.size()];
        final List<IntPair> list = align(x.words(), y.words());
        for (IntPair ip : list) {
            matrix[ip._1][ip._2] = 1;
        }
        return new Alignment(x, y, matrix);
    }

    @Override
    public void save(File file) throws IOException {
    }

    public static class Trainer implements AlignmentTrainer<EasyMonolingualAligner> {

        @Override
        public EasyMonolingualAligner train(List<TrainingPair> data) {
            return new EasyMonolingualAligner();
        }

        @Override
        public EasyMonolingualAligner load(File file) throws IOException {
            return new EasyMonolingualAligner();
        }
    }
}

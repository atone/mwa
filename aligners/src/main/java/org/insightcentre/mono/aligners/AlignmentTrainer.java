package org.insightcentre.mono.aligners;


import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Train an aligner
 * @author jmccrae
 * @param <A> The actual type of aligner
 */
public interface AlignmentTrainer<A extends Aligner> {
    public static class TrainingPair {
        public final SentenceVectors x;
        public final SentenceVectors y;
        public double simScore;

        public TrainingPair(SentenceVectors x, SentenceVectors y) {
            this.x = x;
            this.y = y;
        }
        
        public TrainingPair(SentenceVectors x, SentenceVectors y, Double simScore) {
            this.x = x;
            this.y = y;
            this.simScore = simScore;
        }
    }
    
    /**
     * Train a model
     * @param data The data to train on
     * @return The trained aligner
     */
    A train(List<TrainingPair> data);
     
    /**
     * Load from disk
     * @param file The file containing the model
     * @return The aligner
     * @throws IOException If any of 100 IO things go wrong
     */
    A load(File file) throws IOException;
}

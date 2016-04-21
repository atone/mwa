package org.insightcentre.mono.aligners.nn;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.AlignmentTrainer;

/**
 *
 * @author John McCrae
 */
public class WordSimAlignerTrainer implements AlignmentTrainer {
    private final boolean train;

    public WordSimAlignerTrainer(boolean train) {
        this.train = train;
    }

    @Override
    public Aligner train(List data) {
        return new WordSimAligner(train);
    }

    @Override
    public Aligner load(File file) throws IOException {
        return new WordSimAligner(train);
    }
    
}

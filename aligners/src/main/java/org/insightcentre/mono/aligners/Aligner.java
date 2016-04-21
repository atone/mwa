package org.insightcentre.mono.aligners;


import java.io.File;
import java.io.IOException;

/**
 * An aligner between two sets of words
 * @author John McCrae
 */
public interface Aligner {
    /**
     * Align to sentences
     * @param x The source sentence to align
     * @param y The target sentence to align
     * @return A matrix giving the alignment
     */
    public Alignment align(SentenceVectors x, SentenceVectors y);
    
    /**
     * Save the model to a file
     * @param file The file to save the model to
     */
    public void save(File file) throws IOException;
}

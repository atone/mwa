package org.insightcentre.mono.aligners;

/**
 *
 * @author John McCrae
 */
public interface FeatureExtractor {
    /**
     * Extract Features
     * @param sentence The tokenized sentence
     * @param original The untokenized sentence
     * @return A vector of tokens. Satisfying returnValue.size() = sentence.length and 
     */
    SentenceVectors extractFeatures(String[] sentence, String original);
}

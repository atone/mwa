package org.insightcentre.mono.aligners.featextractors;


import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.SentenceVectors;

import edu.insight.unlp.nn.common.nlp.GloveVectors;
import edu.insight.unlp.nn.common.nlp.Word2Vector;

/**
 * @author Kartik Asooja
 */
public class GloveFeatureExtractor implements FeatureExtractor {

	private static String embeddingPath = "models/glove.6B.50d.txt";
	
	private static Word2Vector w2vec = new GloveVectors(embeddingPath); 
	
    @Override
    public SentenceVectors extractFeatures(String[] sentence, String original) {
        double[][] vectors = new double[sentence.length][];
        for(int i = 0; i < sentence.length; i++) {
        		double[] vector = w2vec.getWordVector(sentence[i].trim().toLowerCase());
        		vectors[i] = vector;
        }
        return new SentenceVectors(vectors, sentence, original);
    }
    
}

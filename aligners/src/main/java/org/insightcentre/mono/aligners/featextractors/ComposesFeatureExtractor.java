package org.insightcentre.mono.aligners.featextractors;


import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.SentenceVectors;

import edu.insight.unlp.nn.common.nlp.ComposesVectors;
import edu.insight.unlp.nn.common.nlp.Word2Vector;

/**
 * @author Kartik Asooja
 */
public class ComposesFeatureExtractor implements FeatureExtractor {

	private static String embeddingPath = "models/EN-wform.w.5.cbow.neg10.400.subsmpl.txt";
	//private static String embeddingPath = "/Users/kartik/Downloads/John/STS/Publications/Distributed:Distributional Vectors/ Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors /Data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt";
	
	private static Word2Vector w2vec = new ComposesVectors(embeddingPath); 
	
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

package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.SentenceVectors;

import static java.lang.Math.sqrt;

/**
 * A word relatedness aligner based on word embeddings.
 * @author Kartik Asooja, John McCrae
 */

public class WordSimAligner implements Aligner {
	private final boolean normalize;

	public WordSimAligner(boolean normalize) {
		this.normalize = normalize;
	}

	public static double cosine(double[] v1, double[] v2) {
		double xx = 0.0, xy = 0.0, yy = 0.0;
		for(int i = 0; i < v1.length; i++) {
			xx += v1[i] * v1[i];
			xy += v1[i] * v2[i];
			yy += v2[i] * v2[i];
		}
		return  xy / sqrt(xx) / sqrt(yy);
	}

	@Override
	public Alignment align(SentenceVectors x, SentenceVectors y) {
		double[][] alignment = new double[x.size()][y.size()];
		for(int i=0; i<x.size(); i++){
			double[] vectorx = x.vector(i);
			double sum = 0;
			for(int j=0; j<y.size(); j++){
				double[] vectory = y.vector(j);
				alignment[i][j] = cosine(vectorx, vectory);
				if(alignment[i][j] == 1 && !x.word(i).equals(y.word(j))) { alignment[i][j] = 0; }
				sum += alignment[i][j] * alignment[i][j];
			}
			if(normalize) {
				sum = sqrt(sum);
				for(int j=0; j<y.size(); j++){
					alignment[i][j] /= sum;
				}
			}

		}
		return new Alignment(x,  y, alignment);		
	}

	@Override
	public void save(File file) throws IOException {
	}

}


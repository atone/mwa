package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.nn.SentenceDataset.SentenceSeqs;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.nlp.Word2VectorUtils;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

/**
 * A Bi-RNN based aligner between two sequences of words.
 * @author Kartik Asooja
 */
public class BRNNSimAligner implements Aligner, Serializable {

	private static final long serialVersionUID = 1L;
	private NN fnn;
	private NN bnn;

	public BRNNSimAligner(NN fnn, NN bnn){
		this.fnn = fnn;		
		this.bnn = bnn;
	}
	
	public NN getFNN(){
		return fnn;
	}
	
	public NN getBNN(){
		return bnn;
	}

	public static double[] concat(double[]... arrays) {
		int length = 0;
		for (double[] array : arrays) {
			length += array.length;
		}
		double[] result = new double[length];
		int pos = 0;
		for (double[] array : arrays) {
			for (double element : array) {
				result[pos] = element;
				pos++;
			}
		}
		return result;
	}

	@Override
	public Alignment align(SentenceVectors x, SentenceVectors y) {
		SentenceSeqs sentxSeqs = SentenceDataset.createSeqs(x);
		SentenceSeqs sentySeqs = SentenceDataset.createSeqs(y);

		SquareErrorFunction ef = new SquareErrorFunction();

		NNLayer fPreOutputLayer = fnn.getLayers().get(fnn.getLayers().size()-2);
		NNLayer bPreOutputLayer = bnn.getLayers().get(bnn.getLayers().size()-2);

		Set<NNLayer> fnnLayersForOutput = new HashSet<NNLayer>();
		fnnLayersForOutput.add(fPreOutputLayer);

		Set<NNLayer> bnnLayersForOutput = new HashSet<NNLayer>();
		bnnLayersForOutput.add(bPreOutputLayer);

		boolean applyTraining = false;

		double[][] xf = fnn.ff(sentxSeqs.forwardSeq, ef, applyTraining, fnnLayersForOutput).get(fPreOutputLayer);
		double[][] yf = fnn.ff(sentySeqs.forwardSeq, ef, applyTraining, fnnLayersForOutput).get(fPreOutputLayer);
		double[][] xb = bnn.ff(sentxSeqs.backwardSeq, ef, applyTraining, bnnLayersForOutput).get(bPreOutputLayer);
		double[][] yb = bnn.ff(sentySeqs.backwardSeq, ef, applyTraining, bnnLayersForOutput).get(bPreOutputLayer);

		double[][] sentxCombinedVecs = new double[x.size()][];

		for(int i=0; i<x.size(); i++){
			sentxCombinedVecs[i] = concat(xf[i], xb[i]);
		}

		double[][] sentyCombinedVecs = new double[y.size()][];

		for(int i=0; i<y.size(); i++){
			sentyCombinedVecs[i] = concat(yf[i], yb[i]);
		}
		
		double[][] alignment = align(sentxCombinedVecs, sentyCombinedVecs);
		return new Alignment(x,  y, alignment);
	}

	private double[][] align(double[][] sentxCombinedVecs, double[][] sentyCombinedVecs){	
		double[][] alignment = new double[sentxCombinedVecs.length][sentyCombinedVecs.length];
		for(int i=0; i<sentxCombinedVecs.length; i++){
			double[] vectorx = sentxCombinedVecs[i];
			for(int j=0; j<sentyCombinedVecs.length; j++){
				double[] vectory = sentyCombinedVecs[j];
				alignment[i][j] = Word2VectorUtils.getSim(vectorx, vectory);
			}
		}
		return alignment;
	}

	@Override
	public void save(File file) throws IOException {
		SerializationUtils.saveObject(this, file);	
	}

}

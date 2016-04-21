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
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

public class NNAligner implements Aligner, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private NN nn;
	private NN fnn, bnn;

	public NNAligner(NN nn, BiLSTMSimAligner aligner){
		this.nn = nn;
		fnn = aligner.getFNN();
		bnn = aligner.getBNN();
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

		double[][] demoOutput = new double[1][];
		demoOutput[0] = new double[]{0.0};

		double[][] alignment = new double[x.size()][y.size()];

		for(int i = 0; i<x.size(); i++){
			for(int j = 0; j<y.size(); j++){
				double[] srcVector = x.vector(i);
				double[] srcBiLSTM = sentxCombinedVecs[i];
				double[] trgVector = y.vector(j);
				double[] trgBiLSTM = sentyCombinedVecs[j];
				double[][] input = new double[1][];
				input[0] = concat(srcVector, srcBiLSTM, trgVector, trgBiLSTM);
				Sequence seq = new Sequence(input, demoOutput);
				double[][] output = nn.ff(seq, ef, applyTraining);
				alignment[i][j]  = output[0][0];
			}
		}
		return new Alignment(x, y, alignment);
	}

	@Override
	public void save(File file) throws IOException {
		SerializationUtils.saveObject(this, file);
	}

}

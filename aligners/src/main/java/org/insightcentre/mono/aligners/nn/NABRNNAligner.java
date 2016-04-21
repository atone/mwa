package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.io.Serializable;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.SentenceVectors;

import edu.insight.unlp.nn.NANN;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.NASequence;
import edu.insight.unlp.nn.common.Sequence;

public class NABRNNAligner implements Aligner, Serializable {

	private static final long serialVersionUID = 1L;
	private NANN naNN;
	private BRNNSimAligner brnnAligner;

	public NABRNNAligner(NN naNN, BRNNSimAligner brnnAligner) {
		this.naNN = (NANN) naNN;
		this.brnnAligner  = brnnAligner;
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
		double[][] alignment = new double[x.size()][y.size()];
		
		Sequence seq = SentencePairBiRNNDataset.createSeqs(x, y, brnnAligner.getFNN(), brnnAligner.getBNN());
		double[][] inputSeq =  ((NASequence) seq).inputSeq;
		double[][] naInput =  ((NASequence) seq).naInput;
		
		for(int i=0; i<inputSeq.length; i++){
			double[] input = seq.inputSeq[i];
			for(int j=-1; j<naInput.length-1; j++){
				double[] naIn = new double[naInput[0].length];
				if(j!=-1){
					naIn = naInput[j];
				}
				
				double[] inputToNa = concat(input, naIn);
				double[] naOutput = naNN.ff(inputToNa, false);
				double naWeight = naOutput[0];
				alignment[i][j+1] = naWeight;
			}
		}
		
		return new Alignment(x,  y, alignment);
	}

	@Override
	public void save(File file) throws IOException {
		SerializationUtils.saveObject(this, file);	
	}

}

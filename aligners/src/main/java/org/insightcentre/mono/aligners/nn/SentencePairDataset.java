package org.insightcentre.mono.aligners.nn;


import java.util.ArrayList;
import java.util.List;

import org.insightcentre.mono.aligners.AlignmentTrainer.TrainingPair;
import org.insightcentre.mono.aligners.SentenceVectors;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

/**
 * Sequence pair dataset for alignment to be used by NNs.
 * @author Kartik Asooja
 */
public class SentencePairDataset extends DataSet {

	private List<TrainingPair> tps; 

	public SentencePairDataset(List<TrainingPair> tps) {
		training = new ArrayList<Sequence>();
		this.tps = tps;
		this.trainingError = new SquareErrorFunction();
		setDataSet();
	}

	@Override
	public String evaluateTest(NN nn) {
		return null;
	}

	public static Sequence createSeqs(SentenceVectors sourceSentenceVector, SentenceVectors targetSentenceVector){
		List<double[]> inputWordVectors = new ArrayList<double[]>();
		List<double[]> outputWordVectors = new ArrayList<double[]>();
		for(int i=0; i<sourceSentenceVector.size(); i++){
			double[] inputVector = sourceSentenceVector.vector(i);
			inputWordVectors.add(inputVector);
		}
		double[][] inputSeq = new double[inputWordVectors.size()][];
		inputSeq = inputWordVectors.toArray(inputSeq);

		for(int i=0; i<targetSentenceVector.size(); i++){
			double[] outputVector = targetSentenceVector.vector(i);
			outputWordVectors.add(outputVector);
		}

		double[][] outputSeq = new double[outputWordVectors.size()][];
		outputSeq = outputWordVectors.toArray(outputSeq);
		Sequence seq = new Sequence(inputSeq, outputSeq);
		return seq;		
	}

	@Override
	public void setDataSet() {
		for(TrainingPair tp : tps){			
			SentenceVectors x = tp.x;
			SentenceVectors y = tp.y;
			Sequence sourceXtargetYSeq = createSeqs(x, y);
			Sequence sourceYtargetXSeq = createSeqs(y, x);
			training.add(sourceXtargetYSeq);
			training.add(sourceYtargetXSeq);			
		}
		setDimensions();
	}

	private void setDimensions(){
		inputUnits = training.get(0).inputSeq[0].length;
		for(Sequence seq : training) {
			if(seq.target!=null){		
				outputUnits = seq.target[0].length;
				break;
			}
		}
	}

}

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
 * Sequence dataset (forward and backward) for sentences to be used by NNs. Language Modeling.
 * @author Kartik Asooja
 */
public class SentenceDataset extends DataSet {

	public static class SentenceSeqs {
		public final Sequence forwardSeq;
		public final Sequence backwardSeq;

		public SentenceSeqs(Sequence forwardSeq, Sequence backwardSeq) {
			this.forwardSeq = forwardSeq;
			this.backwardSeq = backwardSeq;
		}
	}

	public List<Sequence> backwardTraining;
	public List<Sequence> backwardTesting;
	private List<TrainingPair> tps; 

	public SentenceDataset(List<TrainingPair> tps) {
		backwardTraining = new ArrayList<Sequence>();
		training = new ArrayList<Sequence>();
		this.tps = tps;
		this.trainingError = new SquareErrorFunction();
		setDataSet();
	}

	@Override
	public String evaluateTest(NN nn) {
		return null;
	}

	public static SentenceSeqs createSeqs(SentenceVectors sentenceVector){
		List<double[]> inputWordFVectors = new ArrayList<double[]>();
		List<double[]> outputWordFVectors = new ArrayList<double[]>();
		List<double[]> inputWordBVectors = new ArrayList<double[]>();
		List<double[]> outputWordBVectors = new ArrayList<double[]>();
		for(int i=0; i<sentenceVector.size(); i++){
			double[] inputFVector = sentenceVector.vector(i);
			double[] outputFVector = null;
			if(i+1<sentenceVector.size()){
				outputFVector = sentenceVector.vector(i+1);
			} else {
				outputFVector = new double[sentenceVector.vector(0).length]; // end zero padding
			}
			inputWordFVectors.add(inputFVector);
			outputWordFVectors.add(outputFVector);

			double[] inputBVector = sentenceVector.vector(i);
			double[] outputBVector = null;

			if(i-1>=0){
				outputBVector = sentenceVector.vector(i-1);
			} else {
				outputBVector = new double[sentenceVector.vector(0).length]; // start zero padding
			}

			inputWordBVectors.add(inputBVector);
			outputWordBVectors.add(outputBVector);
		}

		double[][] inputFSeq = new double[inputWordFVectors.size()][];
		inputFSeq = inputWordFVectors.toArray(inputFSeq);
		double[][] outputFSeq = new double[outputWordFVectors.size()][];
		outputFSeq = outputWordFVectors.toArray(outputFSeq);
		Sequence forwardSeq = new Sequence(inputFSeq, outputFSeq);

		double[][] inputBSeq = new double[inputWordBVectors.size()][];
		inputBSeq = inputWordBVectors.toArray(inputBSeq);
		double[][] outputBSeq = new double[outputWordBVectors.size()][];
		outputBSeq = outputWordBVectors.toArray(outputBSeq);
		Sequence backwardSeq = new Sequence(inputBSeq, outputBSeq);

		SentenceSeqs sentenceSeqs = new SentenceSeqs(forwardSeq, backwardSeq);
		return sentenceSeqs;		
	}

	public static SentenceSeqs createSeqs(double[][] wordVectors){
		List<double[]> inputWordFVectors = new ArrayList<double[]>();
		List<double[]> outputWordFVectors = new ArrayList<double[]>();
		List<double[]> inputWordBVectors = new ArrayList<double[]>();
		List<double[]> outputWordBVectors = new ArrayList<double[]>();
		for(int i=0; i<wordVectors.length; i++){
			double[] inputFVector = wordVectors[i];
			double[] outputFVector = null;
			if(i+1<wordVectors.length){
				outputFVector = wordVectors[i+1];
			} else {
				outputFVector = new double[wordVectors[0].length]; // end zero padding
			}
			inputWordFVectors.add(inputFVector);
			outputWordFVectors.add(outputFVector);

			double[] inputBVector = wordVectors[i];//sentenceVector.vector(i);
			double[] outputBVector = null;

			if(i-1>=0){
				outputBVector = wordVectors[i-1];//sentenceVector.vector(i-1);
			} else {
				outputBVector = new double[wordVectors[0].length]; // start zero padding
			}

			inputWordBVectors.add(inputBVector);
			outputWordBVectors.add(outputBVector);
		}

		double[][] inputFSeq = new double[inputWordFVectors.size()][];
		inputFSeq = inputWordFVectors.toArray(inputFSeq);
		double[][] outputFSeq = new double[outputWordFVectors.size()][];
		outputFSeq = outputWordFVectors.toArray(outputFSeq);
		Sequence forwardSeq = new Sequence(inputFSeq, outputFSeq);

		double[][] inputBSeq = new double[inputWordBVectors.size()][];
		inputBSeq = inputWordBVectors.toArray(inputBSeq);
		double[][] outputBSeq = new double[outputWordBVectors.size()][];
		outputBSeq = outputWordBVectors.toArray(outputBSeq);
		Sequence backwardSeq = new Sequence(inputBSeq, outputBSeq);

		SentenceSeqs sentenceSeqs = new SentenceSeqs(forwardSeq, backwardSeq);
		return sentenceSeqs;		
	}

	
	@Override
	public void setDataSet() {
		for(TrainingPair tp : tps){			
			SentenceVectors x = tp.x;
			SentenceVectors y = tp.y;
			SentenceSeqs sentxSeqs = createSeqs(x);
			SentenceSeqs sentySeqs = createSeqs(y);
			training.add(sentxSeqs.forwardSeq);
			training.add(sentySeqs.forwardSeq);
			backwardTraining.add(sentxSeqs.backwardSeq);
			backwardTraining.add(sentySeqs.backwardSeq);
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

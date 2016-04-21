package org.insightcentre.mono.aligners.nn;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.AlignmentTrainer.TrainingPair;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.nn.SentenceDataset.SentenceSeqs;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.NASequence;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

/**
 * Sequence pair dataset (input sequence RNNified) for alignment to be used by NNs, NA model.
 * @author Kartik Asooja
 */
public class SentencePairBiRNNDataset extends DataSet {

	private List<TrainingPair> tps; 
	private Aligner aligner;
	
	public SentencePairBiRNNDataset(List<TrainingPair> tps, Aligner aligner) {
		training = new ArrayList<Sequence>();
		this.tps = tps;
		this.trainingError = new SquareErrorFunction();
		this.aligner = aligner;
		setDataSet();
	}

	@Override
	public String evaluateTest(NN nn) {
		return null;
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

	public static Sequence createSeqs(SentenceVectors sourceSentenceVector, SentenceVectors targetSentenceVector, NN fnn, NN bnn){
		List<double[]> inputWordVectors = new ArrayList<double[]>();
		//	List<double[]> outputWordVectors = new ArrayList<double[]>();
		for(int i=0; i<sourceSentenceVector.size(); i++){
			double[] inputVector = sourceSentenceVector.vector(i);
			inputWordVectors.add(inputVector);
		}

		SentenceSeqs sentxSeqs = SentenceDataset.createSeqs(sourceSentenceVector);
		SentenceSeqs sentySeqs = SentenceDataset.createSeqs(targetSentenceVector);

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
		//	double[][] yb = bnn.ff(sentySeqs.backwardSeq, ef, applyTraining, bnnLayersForOutput).get(bPreOutputLayer);

		double[][] sentxCombinedVecs = new double[sourceSentenceVector.size()][];

		for(int i=0; i<sourceSentenceVector.size(); i++){
			sentxCombinedVecs[i] = concat(xf[i], xb[i]);
		}

		//		double[][] sentyCombinedVecs = new double[targetSentenceVector.size()][];
		//
		//		for(int i=0; i<targetSentenceVector.size(); i++){
		//			sentyCombinedVecs[i] = concat(yf[i], yb[i]);
		//		}		


		double[][] inputSeq = sentxCombinedVecs;//new double[inputWordVectors.size()][];

		//inputSeq = inputWordVectors.toArray(inputSeq);

		//		inputSeq = inputWordVectors.toArray(inputSeq);

		//		for(int i=0; i<targetSentenceVector.size(); i++){
		//			double[] outputVector = targetSentenceVector.vector(i);
		//			outputWordVectors.add(outputVector);
		//		}

		double[][] outputSeq = sentySeqs.forwardSeq.inputSeq;//new double[outputWordVectors.size()][];
		//outputSeq = outputWordVectors.toArray(outputSeq);
		Sequence seq = new NASequence(inputSeq, outputSeq, yf);
		return seq;		
	}

	@Override
	public void setDataSet() {
		NN fnn =  null;
		NN bnn = null;
		if(BRNNSimAligner.class.isInstance(aligner)){
			fnn = ((BRNNSimAligner) aligner).getFNN();
			bnn = ((BRNNSimAligner) aligner).getBNN();
		} else if(BiLSTMSimAligner.class.isInstance(aligner)){
			fnn = ((BiLSTMSimAligner) aligner).getFNN();
			bnn = ((BiLSTMSimAligner) aligner).getBNN();
		}

		for(TrainingPair tp : tps){			
			SentenceVectors x = tp.x;
			SentenceVectors y = tp.y;
			Sequence sourceXtargetYSeq = null;
			Sequence sourceYtargetXSeq = null;
			sourceXtargetYSeq = createSeqs(x, y, fnn, bnn);
			sourceYtargetXSeq = createSeqs(y, x, fnn, bnn);
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

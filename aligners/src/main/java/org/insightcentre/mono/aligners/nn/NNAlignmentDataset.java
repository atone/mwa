package org.insightcentre.mono.aligners.nn;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.PrettyGoodTokenizer;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.nn.NNAlignerTrainer.Alignment;
import org.insightcentre.mono.aligners.nn.SentenceDataset.SentenceSeqs;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

public class NNAlignmentDataset extends DataSet {

	private List<Alignment> alignments; 
	private FeatureExtractor fe;
	private NN fnn;
	private NN bnn;

	public NNAlignmentDataset(List<Alignment> alignments, FeatureExtractor fe, BiLSTMSimAligner aligner) {
		training = new ArrayList<Sequence>();
		this.alignments = alignments;
		this.fe = fe;
		this.fnn = aligner.getFNN();
		this.bnn = aligner.getBNN();
		this.trainingError = new SquareErrorFunction();
		setDataSet();
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

	@Override
	public String evaluateTest(NN nn) {
		// TODO Auto-generated method stub
		return null;
	}

	public static double[] concat(double[]... arrays) {
		int length = 0;
		for (double[] array : arrays) {
			length += array.length;
		}
		double[] result = new double[length];
		int pos = 0;

		//int prevArrayLength = 0;
		for (double[] array : arrays) {			
			//	System.arraycopy(array, 0, result, prevArrayLength, array.length);
			//	prevArrayLength = array.length;
			for (double element : array) {
				result[pos] = element;
				pos++;
			}
		}
		return result;
	}

	//	private void collectGarbage() {
	//		for (int i = 0; i < 4; i++) {
	//			System.gc();
	//			try {
	//				Thread.sleep(10);
	//			} catch (InterruptedException e) {
	//				Thread.currentThread().interrupt();
	//				break;
	//			}
	//		}
	//	}

	@Override
	public void setDataSet() {
		//int co = 0;
		int m = 0;
		int pos = 0;
		int neg = 0;
		int totalNegProduced = 0;
		List<Sequence> somePosSequences = new ArrayList<Sequence>();
		for(Alignment alignment : alignments){
			System.out.println(m++);
			if(m>4){
				break;
			}
			//	co++;
			//			if(co>500){
			//				collectGarbage();
			//				co=0;
			//			}
			final String[] src = PrettyGoodTokenizer.tokenize(alignment.sourceSentence.trim());
			final String[] trg = PrettyGoodTokenizer.tokenize(alignment.targetSentence.trim());

			SentenceVectors srcVecs = fe.extractFeatures(src, alignment.sourceSentence);
			SentenceVectors trgVecs = fe.extractFeatures(trg, alignment.targetSentence);

			SentenceSeqs sentxSeqs = SentenceDataset.createSeqs(srcVecs);
			SentenceSeqs sentySeqs = SentenceDataset.createSeqs(trgVecs);

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

			//			double[][] sentxCombinedVecs = new double[srcVecs.size()][];
			//
			//			for(int i=0; i<srcVecs.size(); i++){
			//				sentxCombinedVecs[i] = concat(xf[i], xb[i]);
			//			}
			//
			//			double[][] sentyCombinedVecs = new double[trgVecs.size()][];
			//
			//			for(int i=0; i<trgVecs.size(); i++){
			//				sentyCombinedVecs[i] = concat(yf[i], yb[i]);
			//			}

			LinkedHashMap<String, List<String>> alignMap = alignment.aligns;
			double[] falseOutput = new double[1];
			falseOutput[0] = 0.0;
			double[] trueOutput = new double[1];
			trueOutput[0] = 1.0;

			for(int i = 0; i<srcVecs.size(); i++){
				String srcWord = srcVecs.word(i).toLowerCase();
				for(int j = 0; j<trgVecs.size(); j++){
					String trgWord = trgVecs.word(j).toLowerCase();
					double[] srcVector = srcVecs.vector(i);
					//double[] srcBiLSTM = sentxCombinedVecs[i];
					double[] trgVector = trgVecs.vector(j);
					//double[] trgBiLSTM = sentyCombinedVecs[j];
					double[][] input = new double[1][];
					input[0] = concat(srcVector, xf[i], xb[i], trgVector, yf[j], yb[j]);

					int[] randArray = new Random().ints(1, 0, 5).toArray();

					//boolean isIt = false;
					if(alignMap.containsKey(srcWord.trim())){
						List<String> list = alignMap.get(srcWord.trim());
						if(list.contains(trgWord.toLowerCase())){
							double[][] output = new double[1][];
							output[0] = trueOutput;
							Sequence seq = new Sequence(input, output);
							training.add(seq);
							somePosSequences.add(seq);
							pos++;
						} else {
							totalNegProduced++;
							if(randArray[0]==1){
								double[][] output = new double[1][];
								output[0] = falseOutput;
								Sequence seq = new Sequence(input, output);						
								training.add(seq);						
								neg++;
							}
						}
					} else {
						if(randArray[0]==1){
							totalNegProduced++;
							double[][] output = new double[1][];
							output[0] = falseOutput;
							Sequence seq = new Sequence(input, output);
							training.add(seq);
							neg++;
						}
					}
				}
			}
		}
		System.out.println("training size: " + training.size());
		System.out.println("Pos Sequences: " + pos);
		System.out.println("Total Neg Produced Sequences: " + totalNegProduced);
		System.out.println("Neg Sequences: " + neg);		
		int howManyToAdd = neg-pos;
		double percentage = 100;
		int k = (int)(howManyToAdd*(percentage/100.0f));
		for(int c=0; c<k; c++){
			for(Sequence seq : somePosSequences){
				if(c<k){
					training.add(seq);
					c++;
				} else {
					break;
				}
			}
		}			

		System.out.println("training size: " + training.size());
	}

}

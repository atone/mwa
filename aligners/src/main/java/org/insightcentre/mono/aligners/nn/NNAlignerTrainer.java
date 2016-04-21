package org.insightcentre.mono.aligners.nn;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.AlignmentTrainer;
import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.featextractors.ComposesFeatureExtractor;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNImpl;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.layers.FullyConnectedFFLayer;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class NNAlignerTrainer implements AlignmentTrainer<NNAligner>{

	public class Alignment {
		public String sourceSentence;
		public String targetSentence;
		public LinkedHashMap<String, List<String>> aligns;
		public Alignment(String sourceSentence, String targetSentence, LinkedHashMap<String, List<String>> aligns){
			this.sourceSentence = sourceSentence;
			this.targetSentence = targetSentence;
			this.aligns = aligns;
		}
	}

	private String dataDir = "alignmentData";
	final FeatureExtractor featureExtractor = new ComposesFeatureExtractor();

	public void learnNN(NN nn, DataSet alignmentDataset){	
		// size and no. of layers to be changed according to the word embedding used.
		NNLayer outputLayer = new FullyConnectedFFLayer(alignmentDataset.outputUnits, new Sigmoid(), nn);
		//NNLayer hiddenLayer2 = new FullyConnectedFFLayer(alignmentDataset.inputUnits/4, new Sigmoid(), nn);
		NNLayer hiddenLayer = new FullyConnectedFFLayer(40, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(alignmentDataset.inputUnits, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		//layers.add(hiddenLayer2);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		int epoch = 0;
		int maxEpochs = 10;
		//int maxEpochs = 10;
		while(epoch<maxEpochs) {
			epoch++;
			double trainingError = nn.sgdTrain(alignmentDataset.training, 0.005, true);
			System.err.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
		}
	}

	@Override
	public NNAligner train(List<AlignmentTrainer.TrainingPair> data) {
		try {
			File model = new File("models/nnaligner.model");
			NNAligner aligner = load(model);
			return aligner;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;	
	}

	public NNAligner trainIt() {
		List<Alignment> alignments = new ArrayList<Alignment>();
		for(File dataFile : new File(dataDir).listFiles()){
			if(dataFile.isHidden())
				continue;
			BufferedReader br = BasicFileTools.getBufferedReader(dataFile);
			String line = null;
			try {
				while((line=br.readLine())!=null){
					//##### news-common:2143 #####
					if(line.matches("#####.+#####")){
						//S: Hackett and Rossignol did not know each other and Hackett had no connection to Colby , Doyle said .
						while((line=br.readLine())!=null){
							boolean nextBreak1 = false;
							if(line.matches("S:.*")){
								String sourceSentence = line.replaceAll("S:", " ").trim();
								while((line=br.readLine())!=null){
									boolean nextBreak = false;
									if(line.matches("T:.*")){
										String targetSentence = line.replaceAll("T:", " ").trim();
										LinkedHashMap<String, List<String>> aligns = new LinkedHashMap<String, List<String>>();
										int noOfSpaces = 0;
										while((line=br.readLine())!=null){
											if(line.matches(".+<->.+")){
												String[] srcSplit = line.substring(0, line.indexOf("<->")).trim().split("\\s+");											
												String[] trgSplit = line.substring(line.indexOf("<->")+3, line.length()).trim().split("\\s+");
												for(String srcWord : srcSplit){
													List<String> trgAligns = new ArrayList<String>();
													for(String trgWord : trgSplit){
														trgAligns.add(trgWord.trim().toLowerCase());															
													}
													aligns.put(srcWord.trim(), trgAligns);
												}										
											}
											if(line.trim().equalsIgnoreCase("")){
												noOfSpaces++;
											}
											if(noOfSpaces>=2){
												Alignment alignment = new Alignment(sourceSentence, targetSentence, aligns);
												alignments.add(alignment);
												nextBreak = true;
												break;
											}
										}
									}
									if(nextBreak){
										nextBreak1 = true;
										break;
									}
								}
							}
							if(nextBreak1){
								break;
							}
						}
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		TrainedBiLSTMSimTrainer bilstmTrainer = new TrainedBiLSTMSimTrainer();
		BiLSTMSimAligner biLSTMSimAligner = bilstmTrainer.train(null);

		NNAlignmentDataset dataset = new NNAlignmentDataset(alignments, featureExtractor, biLSTMSimAligner);
		NN nn = new NNImpl(new SquareErrorFunction());
		learnNN(nn, dataset);
		NNAligner aligner = new NNAligner(nn, biLSTMSimAligner);
		return aligner;
	}

	@Override
	public NNAligner load(File file) throws IOException {
		NNAligner nnAligner = SerializationUtils.readObject(file);
		return nnAligner;
	}


	public static void main(String[] args) {
		NNAlignerTrainer trainer = new NNAlignerTrainer();
		NNAligner aligner = trainer.trainIt();
		File model = new File("models/nnaligner.model");
		try {
			aligner.save(model);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

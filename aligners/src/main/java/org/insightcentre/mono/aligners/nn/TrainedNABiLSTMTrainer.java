package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.AlignmentTrainer;

/**
 * Trainer for NA Bi-LSTM based aligner.
 * @author Kartik Asooja
 */
public class TrainedNABiLSTMTrainer implements AlignmentTrainer<NABiLSTMAligner>{

	private final static String SCORE_THRESHOLD_STRING = "05_25000";
	private final static int maxDocs = 7000;

	private String toSerializeModelPath = "models/na" + String.valueOf(maxDocs) + "bilstmComposes_" + SCORE_THRESHOLD_STRING + ".model";
	private static NABiLSTMAligner nabilstmSimAligner = null;

	private File nabilstmModel = new File(toSerializeModelPath);

	public TrainedNABiLSTMTrainer() {
		try {
			nabilstmSimAligner = load(nabilstmModel);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public NABiLSTMAligner train(List<TrainingPair> data) {
		return nabilstmSimAligner;
	}

	@Override
	public NABiLSTMAligner load(File file) throws IOException {
		NABiLSTMAligner nabilstmSimAligner = SerializationUtils.readObject(file);
		return nabilstmSimAligner;
	}
}


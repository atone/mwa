package org.insightcentre.mono.aligners.nn;


import java.io.File;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.AlignmentTrainer;

/**
 * Trainer for Bi-RNN based aligner.
 * @author Kartik Asooja
 */
public class TrainedBRNNSimTrainer implements AlignmentTrainer<BRNNSimAligner>{

	private final static String SCORE_THRESHOLD_STRING = "08_44000";

	private String toSerializeModelPath = "models/" + "birnnComposes_" + SCORE_THRESHOLD_STRING + ".model";
	private static BRNNSimAligner brnnSimAligner = null;

	private File brnnModel = new File(toSerializeModelPath);

	public TrainedBRNNSimTrainer() {
		try {
			brnnSimAligner = load(brnnModel);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public BRNNSimAligner train(List<TrainingPair> data) {
		return brnnSimAligner;
	}

	@Override
	public BRNNSimAligner load(File file) throws IOException {
		BRNNSimAligner brnnSimAligner = SerializationUtils.readObject(file);
		return brnnSimAligner;
	}
}


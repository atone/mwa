package org.insightcentre.mono.aligners.nn;

import static cc.mallet.util.ArrayUtils.indexOf;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.AlignmentTrainer;
import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.PrettyGoodTokenizer;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.featextractors.ComposesFeatureExtractor;
import org.insightcentre.mono.aligners.util.IntPair;

/**
 *
 * @author John McCrae
 */
public class SultanAligner implements AlignmentTrainer<SultanAligner.A> {

	@Override
	public A train(List<TrainingPair> data) {
		return new A("models/sultan-aligns.csv");
	}

	@Override
	public A load(File file) throws IOException {
		return new A("models/sultan-aligns.csv");
	}

	public static class A implements Aligner {
		private final FeatureExtractor featExtract = new ComposesFeatureExtractor();
		private static Map<String, List<IntPair>> map = new HashMap<>();

		public A(String fileName) {
			try(BufferedReader in = new BufferedReader(new FileReader(fileName))) {
				String line;
				while((line = in.readLine()) != null) {
					String[] elems = line.split(" \\|\\|\\| ");
					String[] src = featExtract.extractFeatures(PrettyGoodTokenizer.tokenize(elems[0]), elems[0]).words();
					String[] trg = featExtract.extractFeatures(PrettyGoodTokenizer.tokenize(elems[1]), elems[1]).words();
					String srcString = String.join(" ", src);
					String trgString = String.join(" ", trg);
					String key = srcString + " ||| " + trgString;
					List<IntPair> pairs = new ArrayList<>();
					if(elems.length == 3) {
						for(String s : elems[2].split(" ")) {
							String[] t = s.split("-");
							if(t.length == 2) {
								int i = indexOf(src, t[0]);
								int j = indexOf(trg, t[1]);
								//int i = Arrays.binarySearch(src, t[0]);
								//int j = Arrays.binarySearch(trg, t[1]);
								if(i >=0 && j >= 0) {
									pairs.add(new IntPair(i, j));
								}
							} /*else {
                                System.err.println(s);
                            }*/
						}
					}
					map.put(key, pairs);
				}
			} catch(IOException x) {
				throw new RuntimeException(x);
			}
		}

		@Override
		public Alignment align(SentenceVectors x, SentenceVectors y) {
			String key = String.join(" ", x.words()) + " ||| " + String.join(" ", y.words());
			if(map.containsKey(key)) {
				List<IntPair> list = map.get(key);
				double[][] matrix = new double[x.size()][y.size()];
				for(IntPair ip : list) {
					matrix[ip._1][ip._2] = 1;
				}
				return new Alignment(x, y, matrix);
			} else {
				System.err.println("Lost: " + x + " ||| " + y);
				return new Alignment(x, y, new double[x.size()][y.size()]);
			}
		}

		@Override
		public void save(File file) throws IOException {
		}

	}
}

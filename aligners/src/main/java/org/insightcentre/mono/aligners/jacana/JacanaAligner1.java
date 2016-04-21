package org.insightcentre.mono.aligners.jacana;



import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.HashMap;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import javax.json.JsonValue;

import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.SentenceVectors;

import edu.insight.unlp.nn.utils.BasicFileTools;
import edu.jhu.jacana.align.aligner.FlatAligner;

public class JacanaAligner1 implements Aligner, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static FlatAligner f = new FlatAligner();
	private static String modelFile = "jacData/Edingburgh_RTE2.all_sure.t2s.model";
	private static HashMap<String, String> prevMap = new HashMap<String, String>();
	private static String prevJacsPath = "jacData/jacAlignerMap1.model";

	static {
		//System.setProperty("JACANA_HOME", "/home/karaso/github/unlp-sts2/unlp-sts/jacData");
		System.setProperty("JACANA_HOME", "/Users/kartik/git/unlp-sts/jacData");
		f.initParams(true);
		loadPrevJacs();
	}

	static {		
		f.readModel(modelFile);
	}

	private static void loadPrevJacs(){
		if(prevMap.isEmpty() && new File(prevJacsPath).exists()){
			prevMap = SerializationUtils.readObject(new File(prevJacsPath));
		}
	}

	public static void writePrevJacs(){
		SerializationUtils.saveObject(prevMap, new File(prevJacsPath));
		System.out.println("jacAln1 writte");
	}

	@Override
	public Alignment align(SentenceVectors x, SentenceVectors y) {
		StringBuilder src = new StringBuilder();
		StringBuilder trg = new StringBuilder();

		src.append(x.original()+" ");
		trg.append(y.original()+ " ");

		String tmpFilePath = "jacData/tmp1.txt";
		String tmpOutFilePath = "jacData/tmp1.json";

		String json = null;
		String key = src.toString().trim() + "\t" + trg.toString().trim();
		if(prevMap.containsKey(key)){
			json = prevMap.get(key);			
		} else { 
			BasicFileTools.writeFile(tmpFilePath, key);
			f.decode(tmpFilePath, tmpOutFilePath);		
			BufferedReader bufferedReader = BasicFileTools.getBufferedReader(tmpOutFilePath);
			JsonReader jsonReader = Json.createReader(bufferedReader);
			JsonArray array = jsonReader.readArray();
			JsonObject jsonObject = array.getJsonObject(0);
			json = jsonObject.toString();
			prevMap.put(key, jsonObject.toString());
			jsonReader.close();
		}

		StringReader stringReader = new StringReader(json);
		JsonReader jsonReader = Json.createReader(stringReader);
		JsonObject jsonObject = jsonReader.readObject();

		JsonValue jsonValue = jsonObject.get("sureAlign");
		JsonValue srcS = jsonObject.get("source");
		JsonValue trgS = jsonObject.get("target");

		String[] split = jsonValue.toString().split("\\s+");
		jsonReader.close();

		double[][] align = new double[srcS.toString().split("\\s+").length][trgS.toString().split("\\s+").length];

		if(jsonValue.toString().contains("-")){
			for(String s : split){
				String[] split2 = s.trim().split("-");
				int sourceIndex = Integer.parseInt(split2[0].replaceAll("\"", "").trim());
				int targetIndex = Integer.parseInt(split2[1].replaceAll("\"", "").trim());
				align[sourceIndex][targetIndex] = 1.0;
			}
		}

		Alignment alignment = new Alignment(x, y, align);
		alignment.setSrcWrds(srcS.toString().replaceAll("\"","").trim().split("\\s+"));
		alignment.setTrgWrds(trgS.toString().replaceAll("\"","").trim().split("\\s+"));
		return alignment;
	}

	@Override
	public void save(File file) throws IOException {

	}

}

package org.insightcentre.mono.aligners.main;


public class SentencePair {

	private String s1, s2;
	private double score;
	
	public SentencePair(){
		
	}

	public SentencePair(String s1, String s2, double score){
		this.s1 = s1;
		this.s2 = s2;
		this.score = score;
	}	

	public String getSentence1() {
		return s1;
	}
	public void setSentence1(String sentence1) {
		this.s1 = sentence1;
	}
	public double getSimScore() {
		return score;
	}
	public void setSimScore(double simScore) {
		this.score = simScore;
	}
	public String getSentence2() {
		return s2;
	}
	public void setSentence2(String sentence2) {
		this.s2 = sentence2;
	}

}

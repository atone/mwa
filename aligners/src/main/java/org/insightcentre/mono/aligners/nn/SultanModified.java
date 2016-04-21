package org.insightcentre.mono.aligners.nn;


import com.fasterxml.jackson.databind.ObjectMapper;
import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import it.unimi.dsi.fastutil.ints.IntRBTreeSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import static java.lang.Math.*;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.insightcentre.mono.aligners.Aligner;
import org.insightcentre.mono.aligners.Alignment;
import org.insightcentre.mono.aligners.AlignmentTrainer;
import org.insightcentre.mono.aligners.FeatureExtractor;
import org.insightcentre.mono.aligners.PrettyGoodTokenizer;
import org.insightcentre.mono.aligners.SentenceVectors;
import org.insightcentre.mono.aligners.featextractors.ComposesFeatureExtractor;
import org.insightcentre.mono.aligners.util.IntPair;
import org.insightcentre.mono.aligners.util.StringPair;

/**
 * Modified re-implementation of Sultan
 *
 * @author John McCrae
 */
public class SultanModified implements Aligner {

    private static final int TEXT_WINDOW = 3;
    private static final double PPDB_SIM = 0.9;
    private static final double w = 0.9;
    private static final String CLASSIFIER_FILE = "models/english.all.3class.distsim.crf.ser.gz";
    private static final String POS_CLASSIFIER_FILE = "models/english-bidirectional-distsim.tagger";
    private final Set<StringPair> ppdb;

    public SultanModified() {
        this.ppdb = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader("models/ppdb"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] e = line.split("\t");
                if (e.length == 2) {
                    ppdb.add(new StringPair(e[0], e[1]));
                }
            }
        } catch (IOException x) {
            throw new RuntimeException(x);
        }
    }
    private static AbstractSequenceClassifier<CoreLabel> classifier;
    private static MaxentTagger tagger;
    private static Morphology morpho;
    
    public static String lemma (String s, String t) {
        if(morpho == null) {
            morpho = new Morphology();
        }
        String l = morpho.lemma(s, t).toLowerCase();
        if(!l.equals(s.toLowerCase())) {
            return l;
        }
        if(t.equals("VBG") && l.endsWith("ing")) {
            return l.substring(0,l.length() - 3);
        }
        if((t.equals("VBN") || t.equals("VBN")) && l.endsWith("ed")) {
            return l.substring(0,l.length() - 2);
        }
        if(t.equals("JJR") && l.endsWith("er")) {
            return l.substring(0, l.length() - 2);
        }
        if(t.equals("JJS") && l.endsWith("est")) {
            return l.substring(0, l.length() - 3);
        }
        
       return l;
 
            
 
    }
    
    public static String[] lemmatize(String[] s, String o) {
        if(morpho == null) {
            morpho = new Morphology();
        }
        if (tagger == null) {
            tagger = new MaxentTagger(POS_CLASSIFIER_FILE);
        }
        PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer<>(new StringReader(o),
            new CoreLabelTokenFactory(), "");
        List<CoreLabel> labels = ptbt.tokenize();
        tagger.tagCoreLabels(labels);
        String[] t = new String[s.length];
        for(int i = 0; i < t.length; i++) {
            t[i] = s[i].toLowerCase();
        }
        int j = 0;
        for (CoreLabel word : labels) {
                // Of course Stanford NLP does its one tokenization, if we
            // disagree this is how we recover
            if (s.length != labels.size()) {
                int k = j;
                for (; k < s.length; k++) {
                    if (word.word().equals(s[k])) {
                        break;
                    }
                }
                if (k < s.length) {
                    final String l = lemma(word.word(), word.tag());
                    if(l != null) {
                        t[k] = l.toLowerCase();
                    }
                    j = k + 1;
                }
            } else {
                final String l = lemma(word.word(), word.tag());
                if(l != null) {
                    t[j++] = l.toLowerCase();
                }
            }
        }
        return t;
    }

    public static List<IntPair> ner(String[] s) {
        try {
            if (classifier == null) {
                classifier = CRFClassifier.getClassifier(CLASSIFIER_FILE);
            }
            List<List<CoreLabel>> result = classifier.classify(String.join(" ", s));
            int begin = 0;
            String last = "";
            boolean in = false;
            int j = 0;
            List<IntPair> rval = new ArrayList<>();
            for (CoreLabel word : result.get(0)) {
                String tag = word.get(CoreAnnotations.AnswerAnnotation.class);
                if (tag.equals("O")) {
                    if (in) {
                        rval.add(new IntPair(begin, j));
                    }
                    in = false;
                } else {
                    if (!in) {
                        begin = j;
                        in = true;
                    }
                }
                // Of course Stanford NLP does its one tokenization, if we
                // disagree this is how we recover
                if (s.length != result.get(0).size()) {
                    int k = j;
                    for (; k < s.length; k++) {
                        if (word.word().equals(s[k])) {
                            break;
                        }
                    }
                    if (k < s.length) {
                        j = k + 1;
                    }
                } else {
                    j++;
                }
                last = tag;
            }
            if (in) {
                rval.add(new IntPair(begin, s.length));
            }
            return rval;
        } catch (IOException | ClassNotFoundException x) {
            throw new RuntimeException(x);
        }
    }

    public static class DepTree {

        public final Map<String, DepTree> map;
        public final String node;
        public final String tag;
        public final int idx;

        public DepTree(Map<String, DepTree> map, String node, String tag, int idx) {
            this.map = map;
            this.node = node;
            this.tag = tag;
            this.idx = idx;
        }

        public DepTree(TypedDependency root, Collection<TypedDependency> tds,
            List<? extends HasWord> sentence, int[] remapping, IntSet stack) {
            this.map = new HashMap<>();
            int t = root.dep().index();
            node = sentence.get(t - 1).word();
            tag = root.dep().tag();
            this.idx = remapping[t - 1];
            if (!stack.contains(t)) {
                IntSet stack2 = new IntRBTreeSet(stack);
                stack2.add(t);
                for (TypedDependency td : tds) {
                    if (td.gov().index() == t && td.dep().index() != t) {
                        map.put(td.reln().getShortName(), new DepTree(td, tds, sentence, remapping, stack2));
                    }
                }
            }
        }

        public DepTree find(int j) {
            if (idx == j) {
                return this;
            } else {
                for (DepTree dt : map.values()) {
                    DepTree dt2 = dt.find(j);
                    if (dt2 != null) {
                        return dt2;
                    }
                }
                return null;
            }
        }

        public List<DepTree> referencing(DepTree target) {
            if (map.values().contains(target)) {
                return Collections.singletonList(this);
            } else {
                List<DepTree> list = new ArrayList<>();
                for (DepTree dt : map.values()) {
                    list.addAll(dt.referencing(target));
                }
                return list;
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("(");
            sb.append(node);
            sb.append("/");
            sb.append(tag);
            if (!map.isEmpty()) {
                sb.append(" ");
            }
            for (Map.Entry<String, DepTree> e : map.entrySet()) {
                sb.append(e.getKey());
                sb.append("=");
                sb.append(e.getValue().toString());
                sb.append(" ");
            }
            if (sb.charAt(sb.length() - 1) == ' ') {
                sb.deleteCharAt(sb.length() - 1);
            }
            sb.append(")");
            return sb.toString();
        }

        @Override
        public int hashCode() {
            int hash = 7;
            hash = 89 * hash + Objects.hashCode(this.map);
            hash = 89 * hash + Objects.hashCode(this.node);
            hash = 89 * hash + Objects.hashCode(this.tag);
            return hash;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null) {
                return false;
            }
            if (getClass() != obj.getClass()) {
                return false;
            }
            final DepTree other = (DepTree) obj;
            if (!Objects.equals(this.map, other.map)) {
                return false;
            }
            if (!Objects.equals(this.node, other.node)) {
                return false;
            }
            if (!Objects.equals(this.tag, other.tag)) {
                return false;
            }
            return true;
        }

    }

    private static LexicalizedParser lp;

    private static int[] remap(List<? extends HasWord> sentence, String[] s) {
        int j = 0;
        int[] map = new int[sentence.size()];
        for (int i = 0; i < sentence.size(); i++) {
            map[i] = j;
            int k = j + 1;
            for (; k < s.length; k++) {
                if (i + 1 < sentence.size() && sentence.get(i + 1).word().equals(s[k])) {
                    break;
                }
            }
            if (k < s.length) {
                j = k;
            }
        }
        return map;
    }

    public static DepTree parse(String[] s) {
        if (lp == null) {
            lp = LexicalizedParser.loadModel("models/englishPCFG.ser.gz");
        }
        TreebankLanguagePack tlp = lp.getOp().langpack();
        Tokenizer<? extends HasWord> toke = tlp.getTokenizerFactory().getTokenizer(new StringReader(String.join(" ", s)));
        List<? extends HasWord> sentence = toke.tokenize();
        int[] remapping = remap(sentence, s);
        Tree parse = lp.parse(sentence);
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        Collection<TypedDependency> collection = gs.typedDependenciesCCprocessed();
        TypedDependency root = null;
        for (TypedDependency td : collection) {
            if (td.gov().index() == 0) {
                root = td;
            }
        }
        return new DepTree(root, collection, sentence, remapping, new IntRBTreeSet());
    }

    public List<IntPair> depContext(DepTree s, DepTree t, int i, int j) {
        if (wordSim(s.node, t.node) > 0) {
            List<IntPair> context = new ArrayList<>();
            DepTree si = s.find(i);
            DepTree tj = t.find(j);
            if (si != null && tj != null && si.tag.charAt(0) == tj.tag.charAt(0)) {
                for (Map.Entry<String, DepTree> sk : s.map.entrySet()) {
                    for (Map.Entry<String, DepTree> tl : t.map.entrySet()) {
                        if (sk.getValue().tag.charAt(0) == tl.getValue().tag.charAt(0)) {
                            DepStruct q1 = new DepStruct(si.tag.charAt(0),
                                sk.getValue().tag.charAt(0),
                                sk.getKey(), tl.getKey(), true);
                            if (depStructs.contains(q1)) {
                                context.add(new IntPair(sk.getValue().idx, tl.getValue().idx));
                            } else {
                                DepStruct q2 = new DepStruct(si.tag.charAt(0),
                                    sk.getValue().tag.charAt(0),
                                    sk.getKey(), "", true);
                                if (depStructs.contains(q2)) {
                                    context.add(new IntPair(sk.getValue().idx, tl.getValue().idx));
                                }
                            }

                        }
                    }
                    for (DepTree dt : t.referencing(tj)) {
                        for (Map.Entry<String, DepTree> tl : dt.map.entrySet()) {
                            if (tl.getValue() == tj && dt.tag.charAt(0) == sk.getValue().tag.charAt(0)) {
                                DepStruct q1 = new DepStruct(si.tag.charAt(0),
                                    sk.getValue().tag.charAt(0),
                                    sk.getKey(), tl.getKey(), false);
                                if (depStructs.contains(q1)) {
                                    context.add(new IntPair(sk.getValue().idx, dt.idx));
                                } else {
                                    DepStruct q2 = new DepStruct(si.tag.charAt(0),
                                        sk.getValue().tag.charAt(0),
                                        sk.getKey(), "", false);
                                    if (depStructs.contains(q2)) {
                                        context.add(new IntPair(sk.getValue().idx, dt.idx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return context;
        } else {
            return Collections.EMPTY_LIST;
        }
    }

    private static class DepStruct {

        public final char srcPos;
        public final char trgPos;
        public final String srcDep;
        public final String trgDep;
        public final boolean orientation;

        public DepStruct(char srcPos, char trgPos, String srcDep, String trgDep, boolean orientation) {
            this.srcPos = srcPos;
            this.trgPos = trgPos;
            this.srcDep = srcDep;
            this.trgDep = trgDep;
            this.orientation = orientation;
        }

        @Override
        public int hashCode() {
            int hash = 7;
            hash = 89 * hash + this.srcPos;
            hash = 89 * hash + this.trgPos;
            hash = 89 * hash + Objects.hashCode(this.srcDep);
            hash = 89 * hash + Objects.hashCode(this.trgDep);
            hash = 89 * hash + (this.orientation ? 1 : 0);
            return hash;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null) {
                return false;
            }
            if (getClass() != obj.getClass()) {
                return false;
            }
            final DepStruct other = (DepStruct) obj;
            if (this.srcPos != other.srcPos) {
                return false;
            }
            if (this.trgPos != other.trgPos) {
                return false;
            }
            if (!Objects.equals(this.srcDep, other.srcDep)) {
                return false;
            }
            if (!Objects.equals(this.trgDep, other.trgDep)) {
                return false;
            }
            if (this.orientation != other.orientation) {
                return false;
            }
            return true;
        }

    }

    private static final Set<DepStruct> depStructs = new HashSet<>(Arrays.asList(new DepStruct[]{
        new DepStruct('V', 'V', "purpcl", "", true),
        new DepStruct('V', 'V', "xcomp", "", true),
        new DepStruct('V', 'N', "agent", "", true),
        new DepStruct('V', 'N', "nsubj", "", true),
        new DepStruct('V', 'N', "xsubj", "", true),
        new DepStruct('V', 'N', "dobj", "", true),
        new DepStruct('V', 'N', "nsubjpass", "", true),
        new DepStruct('V', 'N', "rel", "", true),
        new DepStruct('V', 'N', "tmod", "", true),
        new DepStruct('V', 'N', "prep_in", "", true),
        new DepStruct('V', 'N', "prep_at", "", true),
        new DepStruct('V', 'N', "prop_on", "", true),
        new DepStruct('V', 'N', "iobj", "", true),
        new DepStruct('V', 'N', "prep_to", "", true),
        new DepStruct('N', 'V', "infmod", "", true),
        new DepStruct('N', 'V', "partmod", "", true),
        new DepStruct('N', 'V', "rcmod", "", true),
        new DepStruct('N', 'N', "pos", "", true),
        new DepStruct('N', 'N', "nn", "", true),
        new DepStruct('N', 'N', "prep_of", "", true),
        new DepStruct('N', 'N', "prep_in", "", true),
        new DepStruct('N', 'N', "prep_at", "", true),
        new DepStruct('N', 'N', "prep_for", "", true),
        new DepStruct('N', 'J', "amod", "", true),
        new DepStruct('N', 'J', "rcmod", "", true),
        new DepStruct('V', 'V', "conj_and", "", false),
        new DepStruct('V', 'V', "conj_or", "", false),
        new DepStruct('V', 'V', "conj_nor", "", false),
        new DepStruct('V', 'N', "dobj", "infmod", false),
        new DepStruct('V', 'N', "dobj", "partmod", false),
        new DepStruct('V', 'N', "dobj", "rcmod", false),
        new DepStruct('V', 'N', "nsubjpass", "infmod", false),
        new DepStruct('V', 'N', "nsubjpass", "partmod", false),
        new DepStruct('V', 'N', "nsubjpass", "rcmod", false),
        new DepStruct('V', 'N', "rel", "infmod", false),
        new DepStruct('V', 'N', "rel", "partmod", false),
        new DepStruct('V', 'N', "rel", "rcmod", false),
        new DepStruct('V', 'J', "acomp", "cop", false),
        new DepStruct('V', 'J', "acomp", "csubj", false),
        new DepStruct('N', 'N', "conj_and", "", false),
        new DepStruct('N', 'N', "conj_or", "", false),
        new DepStruct('N', 'N', "conj_nor", "", false),
        new DepStruct('N', 'J', "amod", "nsubj", false),
        new DepStruct('N', 'J', "rcmod", "nsubj", false),
        new DepStruct('J', 'J', "conj_and", "", false),
        new DepStruct('J', 'J', "conj_or", "", false),
        new DepStruct('J', 'J', "conj_nor", "", false),
        new DepStruct('R', 'R', "conj_and", "", false),
        new DepStruct('R', 'R', "conj_or", "", false),
        new DepStruct('R', 'R', "conj_nor", "", false),}));

    private static int minimum(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }

    public static int levenshtein(CharSequence lhs, CharSequence rhs) {
        int[][] distance = new int[lhs.length() + 1][rhs.length() + 1];

        for (int i = 0; i <= lhs.length(); i++) {
            distance[i][0] = i;
        }
        for (int j = 1; j <= rhs.length(); j++) {
            distance[0][j] = j;
        }

        for (int i = 1; i <= lhs.length(); i++) {
            for (int j = 1; j <= rhs.length(); j++) {
                distance[i][j] = minimum(
                    distance[i - 1][j] + 1,
                    distance[i][j - 1] + 1,
                    distance[i - 1][j - 1] + ((lhs.charAt(i - 1) == rhs.charAt(j - 1)) ? 0 : 1));
            }
        }

        return distance[lhs.length()][rhs.length()];
    }

    public String acronymize(String[] s) {
        final StringBuilder sb = new StringBuilder();
        for (String t : s) {
            if (t != null && t.length() > 0) {
                sb.append(t.charAt(0));
            }
        }
        return sb.toString();
    }

    public boolean isAcronym(String[] s, String[] t) {
        String sacr = acronymize(s);
        String tacr = acronymize(t);
        String sfull = String.join(" ", s);
        String tfull = String.join(" ", t);

        return (levenshtein(sacr, tfull) <= tfull.length() / 4)
            || (levenshtein(tacr, sfull) <= sfull.length() / 4)
            || (levenshtein(sacr, tacr) <= min(sacr.length(), tacr.length()) / 4);
    }

    public List<IntPair> neAlign(String[] s, String[] t, List<IntPair> a) {
        List<IntPair> sNers = ner(s);
        List<IntPair> tNers = ner(t);

        for (IntPair sNer : sNers) {
            String[] sNerStr = Arrays.copyOfRange(s, sNer._1, sNer._2);
            for (IntPair tNer : tNers) {
                String[] tNerStr = Arrays.copyOfRange(t, tNer._1, tNer._2);
                if (isAcronym(sNerStr, tNerStr)) {
                    for (int i = sNer._1; i < sNer._2; i++) {
                        a.add(new IntPair(i, tNer._1 + ((i - sNer._1) * (tNer._2 - tNer._1) / (sNer._2 - sNer._1))));
                    }
                } else {
                    boolean hasAlign = false;
                    for (IntPair ip : a) {
                        hasAlign = hasAlign || (ip._1 >= sNer._1 && ip._1 < sNer._2
                            && ip._2 >= tNer._1 && ip._2 < tNer._2);
                    }
                    if (hasAlign) {
                        if (sNer._2 - sNer._1 > tNer._2 - tNer._1) {
                            for (int i = sNer._1; i < sNer._2; i++) {
                                boolean i_exists = false;
                                for (IntPair ip : a) {
                                    i_exists = i_exists || (ip._1 == i);
                                }
                                if (!i_exists) {
                                    a.add(new IntPair(i, tNer._1 + ((i - sNer._1) * (tNer._2 - tNer._1) / (sNer._2 - sNer._1))));
                                }
                            }
                        } else {
                            for (int j = tNer._1; j < tNer._2; j++) {
                                boolean j_exists = false;
                                for (IntPair ip : a) {
                                    j_exists = j_exists || (ip._2 == j);
                                }
                                if (!j_exists) {
                                    a.add(new IntPair(sNer._1 + ((j - tNer._1) * (sNer._2 - sNer._1) / (tNer._2 - tNer._1)), j));
                                }
                            }

                        }
                    }
                }
            }
        }
        return a;

    }

    public static List<IntPair> textContext(String[] s, String[] t, int i, int j) {
        return new AbstractList<IntPair>() {
            final int i_min = max(0, i - TEXT_WINDOW);
            final int i_max = min(s.length, i + TEXT_WINDOW + 1);
            final int j_min = max(0, j - TEXT_WINDOW);
            final int j_max = min(t.length, j + TEXT_WINDOW + 1);

            @Override
            public IntPair get(int k) {
                int i_off = k / (j_max - j_min) + i_min - i;
                int j_off = k % (j_max - j_min) + j_min - j;
                if (i_off < 0 || (i_off == 0 && j_off < 0)) {
                    return new IntPair(k / (j_max - j_min) + i_min,
                        k % (j_max - j_min) + j_min);
                } else {
                    return new IntPair((k + 1) / (j_max - j_min) + i_min,
                        (k + 1) % (j_max - j_min) + j_min);
                }
            }

            @Override
            public int size() {
                return (i_max - i_min) * (j_max - j_min) - 1;
            }
        };
    }

    public double wordSim(String w1, String w2) {
        if (w1.equals(w2)) {
            return 1.0;
        } else if (ppdb.contains(new StringPair(w1, w2))) {
            return PPDB_SIM;
        } else {
            return 0.0;
        }
    }

    public List<IntPair> wsAlign(String[] s, String[] t) {
        List<IntPair> a = new ArrayList<>();
        for (int i = 0; i < s.length; i++) {
            for (int j = 0; j < t.length; j++) {
                if (s[i].equals(t[j])) {
                    a.add(new IntPair(i, j));
                    break;
                }
            }
        }
        return a;
    }

    public List<IntPair> cwDepAlign(String[] s, String[] t,
        List<IntPair> a_e) {
        final DepTree sTree = parse(s);
        final DepTree tTree = parse(t);
        final List<IntPair> psi = new ArrayList<>();
        final Object2DoubleMap<IntPair> phi = new Object2DoubleOpenHashMap<>();
        final List<IntPair> A = new ArrayList<>();
        A.addAll(a_e);

        for (int i = 0; i < s.length; i++) {
            for (int j = 0; j < t.length; j++) {
                boolean exists_l = false, exists_k = false;
                for (IntPair ip : a_e) {
                    exists_l = exists_l || (ip._1 == i);
                    exists_k = exists_k || (ip._2 == j);
                }
                final double ws = wordSim(s[i], t[j]);
                if (!exists_l && !exists_k && ws > 0) {
                    final IntPair ij = new IntPair(i, j);
                    psi.add(ij);
                    final List<IntPair> context = depContext(sTree, tTree, i, j);
                    double contextSim = 0.0;
                    for (IntPair ip : context) {
                        contextSim += wordSim(s[ip._1], t[ip._2]);
                    }
                    phi.put(ij, w * ws + (1 - w) * contextSim);
                }
            }
        }

        psi.sort(new Comparator<IntPair>() {

            @Override
            public int compare(IntPair t, IntPair t1) {
                final int i = -Double.compare(phi.getDouble(t), phi.getDouble(t1));
                if (i == 0) {
                    final int j = Integer.compare(t._1, t1._1);
                    if (j == 0) {
                        return Integer.compare(t._2, t1._2);
                    }
                    return j;
                }
                return i;
            }
        });

        for (IntPair ip : psi) {
            int i = ip._1;
            int j = ip._2;
            boolean exists_l = false, exists_k = false;
            for (IntPair ip2 : A) {
                exists_l = exists_l || (ip2._1 == i);
                exists_k = exists_k || (ip2._2 == j);
            }
            if (!exists_l && !exists_k) {
                A.add(ip);
            }
        }
        return A;
    }

    public List<IntPair> cwTextAlign(String[] s, String[] t,
        List<IntPair> a_e) {
        final List<IntPair> psi = new ArrayList<>();
        final Object2DoubleMap<IntPair> phi = new Object2DoubleOpenHashMap<>();
        final List<IntPair> A = new ArrayList<>();
        A.addAll(a_e);

        for (int i = 0; i < s.length; i++) {
            for (int j = 0; j < t.length; j++) {
                boolean exists_l = false, exists_k = false;
                for (IntPair ip : a_e) {
                    exists_l = exists_l || (ip._1 == i);
                    exists_k = exists_k || (ip._2 == j);
                }
                final double ws = wordSim(s[i], t[j]);
                if (!exists_l && !exists_k && ws > 0) {
                    final IntPair ij = new IntPair(i, j);
                    psi.add(ij);
                    final List<IntPair> context = textContext(s, t, i, j);
                    double contextSim = 0.0;
                    for (IntPair ip : context) {
                        contextSim += wordSim(s[ip._1], t[ip._2]);
                    }
                    phi.put(ij, w * ws + (1 - w) * contextSim);
                }
            }
        }

        psi.sort(new Comparator<IntPair>() {

            @Override
            public int compare(IntPair t, IntPair t1) {
                final int i = -Double.compare(phi.getDouble(t), phi.getDouble(t1));
                if (i == 0) {
                    final int j = Integer.compare(t._1, t1._1);
                    if (j == 0) {
                        return Integer.compare(t._2, t1._2);
                    }
                    return j;
                }
                return i;
            }
        });

        for (IntPair ip : psi) {
            int i = ip._1;
            int j = ip._2;
            boolean exists_l = false, exists_k = false;
            for (IntPair ip2 : A) {
                exists_l = exists_l || (ip2._1 == i);
                exists_k = exists_k || (ip2._2 == j);
            }
            if (!exists_l && !exists_k) {
                A.add(ip);
            }
        }
        return A;
    }

    public List<IntPair> align(String[] s, String[] t, String os, String ts) {
        return cwTextAlign(s, t, neAlign(s, t, wsAlign(lemmatize(s, os), lemmatize(t, ts))));
//        return cwDepAlign(s, t, cwTextAlign(s, t, neAlign(s, t, wsAlign(s, t))));
    }

    @Override
    public Alignment align(SentenceVectors x, SentenceVectors y) {
        double[][] matrix = new double[x.size()][y.size()];
        final List<IntPair> list = align(x.words(), y.words(), x.original(), y.original());
        for (IntPair ip : list) {
            matrix[ip._1][ip._2] = 1;
        }
        return new Alignment(x, y, matrix);
    }

    @Override
    public void save(File file) throws IOException {
    }

    public static class Trainer implements AlignmentTrainer<SultanModified> {

        @Override
        public SultanModified train(List<TrainingPair> data) {
            return new SultanModified();
        }

        @Override
        public SultanModified load(File file) throws IOException {
            return new SultanModified();
        }
    }

    private static String fixedString(String s, int i) {
        if (s.length() > i) {
            return s.substring(0, i);
        } else {
            return String.format("%-" + i + "s", s);
        }
    }

    private static void printDiff(String[] s1, String[] s2, int n, int m, char symbol) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s1.length; i++) {
            if (i == n) {
                sb.append(symbol).append(fixedString(s1[i], 10));
            } else {
                sb.append(" ").append(fixedString(s1[i], 10));
            }
        }
        sb.append("\n");
        for (int j = 0; j < s2.length; j++) {
            if (j == m) {
                sb.append(symbol).append(fixedString(s2[j], 10));
            } else {
                sb.append(" ").append(fixedString(s2[j], 10));
            }
        }
        System.out.println(sb.toString());
        System.out.println();
    }

    public static void main(String[] args) throws Exception {
        final File dataFile = new File("data/all.json");
        final ObjectMapper mapper = new ObjectMapper();
        final Map<String, List<Map<String, Object>>> data = mapper.readValue(dataFile, Map.class);

        final SultanAligner.A sultanAligner = new SultanAligner().train(new ArrayList<>());
        //final SultanModified sm = new SultanModified();
        final EasyMonolingualAligner eam = new EasyMonolingualAligner();
        final FeatureExtractor featExtract = new ComposesFeatureExtractor();
        double precision = 0;
        double recall = 0;
        int n = 0;
        for (List<Map<String, Object>> dataset : data.values()) {
            for (Map<String, Object> target : dataset) {
                String[] s1 = PrettyGoodTokenizer.tokenize(target.get("s1").toString());
                String[] s2 = PrettyGoodTokenizer.tokenize(target.get("s2").toString());

                SentenceVectors sv1 = featExtract.extractFeatures(s1, target.get("s1").toString());
                SentenceVectors sv2 = featExtract.extractFeatures(s2, target.get("s2").toString());

                Alignment a1 = sultanAligner.align(sv1, sv2);
                Alignment a2 = eam.align(sv1, sv2);
                int tp = 0, fp = 0, fn = 0;
                boolean sultanfail = true;
                for (int i = 0; i < sv1.size(); i++) {
                    for (int j = 0; j < sv2.size(); j++) {
                        if (a1.alignment(i, j) == 1 && a2.alignment(i, j) == 1) {
                            tp++;
                            sultanfail = false;
                        } else if (a1.alignment(i, j) == 1) {
                            fn++;
                            sultanfail = false;
                            printDiff(sv1.words(), sv2.words(), i, j, '#');
                        } else if (a2.alignment(i, j) == 1) {
                            fp++;
                            printDiff(sv1.words(), sv2.words(), i, j, '*');
                        }
                    }
                }
                if (!sultanfail && (tp > 0 || (fp > 0 && fn > 0))) {
                    precision += (double) tp / (tp + fp);
                    recall += (double) tp / (tp + fn);
                    n++;
                }
            }
        }
        System.out.println(String.format("Precision: % .8f", precision / n));
        System.out.println(String.format("Recall   : % .8f", recall / n));
        System.out.println(String.format("Size     : %d", n));
    }
}

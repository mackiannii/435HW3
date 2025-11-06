package org.example;

import java.util.*;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;

public class PageRank {
    public static void main(String[] args){

        if(args.length<3){
            System.exit(1);
        }

        final String links = args[0];
        final String titles = args[1];
        final String output = args[2];
        final int iters = (args.length >= 4) ? Integer.parseInt(args[3]) : 25;

        SparkConf conf = new SparkConf().setAppName("PA3").setMaster("yarn");

        JavaSparkContext sc =  new JavaSparkContext(conf);

        // Loading titles


        JavaRDD<String> titlesRdd = sc.textFile(titles);
        long N = titlesRdd.count(); // total no of pages

        JavaPairRDD<Integer, String> p_titles = titlesRdd.zipWithIndex().mapToPair(new PairFunction<Tuple2<String, Long>, Integer, String>() {
            
            public Tuple2<Integer, String> call(Tuple2<String, Long> t) {
                return new Tuple2<>((int)(t._2 + 1), t._1); // id, title
            }
        });


        // links

        JavaRDD<String> p_links = sc.textFile(links);

        JavaPairRDD<Integer, List<Integer>> link = p_links.filter(new Function<String, Boolean>() {
            
            public Boolean call(String line) {
                return line.contains(":");
            }

            
        }).mapToPair(new PairFunction<String, Integer, List<Integer>>() {
            
            public Tuple2<Integer, List<Integer>> call(String line) {
                String[] parts = line.split(":",2);


                int f_id = Integer.parseInt(parts[0].trim()); //from id's
                List<Integer> t_ids = new ArrayList<>(); //to id's
                if (parts.length>1) {
                    String right = parts[1].trim();
                    if(!right.isEmpty()){
                        for (String ids:right.split("\\s+")) {
                            t_ids.add(Integer.parseInt(ids));
                        }
                    }
                }
                return new Tuple2<>(f_id, t_ids);
            }

            // merging duplicate lists from_ids

        }).reduceByKey(new Function2<List<Integer>, List<Integer>, List<Integer>>() {

            public List<Integer> call(List<Integer> a, List<Integer> b) {
                ArrayList<Integer> m = new ArrayList<>(a.size()+b.size());
                m.addAll(a);
                m.addAll(b);
                return m;
            }
        });

         // Page Id's
        // Initial Ranks


        JavaRDD<Integer> from_ids = link.keys();
        JavaRDD<Integer> to_ids = link.values().flatMap(new FlatMapFunction<List<Integer>, Integer>() {
            
            public Iterator<Integer> call(List<Integer> list) {
                return list.iterator();
            }
        });

        
        JavaRDD<Integer> all_ids = from_ids.union(to_ids).distinct();

        final double ir; // initial ranks for pages
        if (N>0) {
             ir = 1.0 / (double) N;
        }
        else{
            ir = 0.0;
        }

        JavaPairRDD<Integer, Double> ranks = all_ids.mapToPair(new PairFunction<Integer, Integer, Double>() { //page -> id,rank
            
            public Tuple2<Integer, Double> call(Integer id) {
                return new Tuple2<>(id, ir);
            }
        });

        // iterations-idealized


        for (int i = 0;i<iters;i++) {
            JavaPairRDD<Integer, Tuple2<List<Integer>, Double>> neighbors = link.join(ranks);

            JavaPairRDD<Integer, Double> individual = neighbors.flatMapToPair(new PairFlatMapFunction<Tuple2<Integer, Tuple2<List<Integer>, Double>>, Integer, Double>() {
                
                public Iterator<Tuple2<Integer, Double>> call(Tuple2<Integer, Tuple2<List<Integer>, Double>> t) {


                    List<Integer> n_list  = t._2._1; // neighbors 1-> [2,3],
                    double rank = t._2._2;
                    if (n_list ==  null || n_list.isEmpty()) { // pages, no out links
                        return Collections.<Tuple2<Integer, Double>>emptyList().iterator();
                    }


                    double equal_value = rank/n_list.size();


                    ArrayList<Tuple2<Integer, Double>> out = new ArrayList<>(n_list.size());

                    for (int j:n_list) out.add(new Tuple2<>(j, equal_value));
                    return out.iterator();
                }
            });

            JavaPairRDD<Integer, Double> new_ranks = individual.reduceByKey(new Function2<Double, Double, Double>() {
                
                public Double call(Double a, Double b) {
                    return a + b;
                }
            });

            JavaPairRDD<Integer, Double> edges = all_ids.mapToPair(new PairFunction<Integer, Integer, Double>() {
                
                public Tuple2<Integer, Double> call(Integer id) {
                    return new Tuple2<>(id, 0.0);
                }
            });

            ranks = new_ranks.union(edges).reduceByKey(new Function2<Double, Double, Double>() {
                
                public Double call(Double a, Double b) {
                    return a + b;
                }
            });
        }

        // joining ranks with titles

        JavaPairRDD<Integer, Tuple2<Double, String>> w_titles = ranks.join(p_titles);

        JavaPairRDD<Double, String> Desc = w_titles.mapToPair(new PairFunction<Tuple2<Integer, Tuple2<Double, String>>, Double, String>() {
            
            public Tuple2<Double, String> call(Tuple2<Integer, Tuple2<Double, String>> t) {
                return new Tuple2<>(t._2._1, t._2._2);
            }
        }).sortByKey(false);

        Desc.map(new Function<Tuple2<Double, String>, String>() {
            
            public String call(Tuple2<Double, String> t) {
                return t._2 + "\t" + String.format("%.8f", t._1);
            }
        }).saveAsTextFile(output + "/all");

        sc.stop();
    }
}

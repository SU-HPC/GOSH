#include <iostream>
#include "argparse.hpp"

using namespace std;


int main(int argc, const char** argv) {

  // make a new ArgumentParser
  ArgumentParser parser;

  // add some arguments to search for
  parser.addArgument("--five", 5);
  parser.addArgument("-c", "--cactus", 1);
  parser.addArgument("-v", "--variadic", '+');
  parser.addFinalArgument("final");

  // parse the command-line arguments - throws if invalid format
  parser.parse(argc, argv);

  if(parser.gotArgument("five")) {
      vector<float> five = parser.retrieve<vector<float> >("five");
      cout << "five: " << five << endl;
  }

  if(parser.gotArgument("cactus")) {
      int cactus = parser.retrieve<int>("cactus");
      cout << "cactus: " << cactus << endl;
  }

  if(parser.gotArgument("variadic")) {
      vector<vector<int> > variadic = parser.retrieve<vector<vector<int> > >("variadic");
      cout << "variadic: " << variadic << endl;
  }

  if(parser.gotArgument("final")) {
      string final = parser.retrieve<string>(final);
      cout << "final: " << final << endl;
  }
}


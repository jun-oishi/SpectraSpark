#include "util.hpp"

namespace util {

using namespace std;

using lu = long unsigned int;

template <typename T>
void loadtxt(const string &filename, vector<vector<T>> &data,
             const string comments, const char delimiter) {
  ifstream ifs(filename);
  if (!ifs) {
    throw runtime_error("File not found: " + filename);
  }
  string line;
  int len_comments = comments.size();
  while (getline(ifs, line)) {
    if (line.substr(0, len_comments) == comments) {
      continue;
    }
    istringstream iss(line);
    vector<T> row;
    string cell;
    while (getline(iss, cell, delimiter)) {
      row.push_back(static_cast<T>(stod(cell)));
    }
    data.push_back(row);
  }
}

template void loadtxt<int>(const std::string &filename,
                           std::vector<std::vector<int>> &data,
                           const std::string comments, const char delimiter);

template void loadtxt<float>(const std::string &filename,
                             std::vector<std::vector<float>> &data,
                             const std::string comments, const char delimiter);

template void loadtxt<double>(const std::string &filename,
                              std::vector<std::vector<double>> &data,
                              const std::string comments, const char delimiter);

template <typename T>
void savetxt(const string &filename, const vector<vector<T>> &data,
             const string header, const string comments, const char delimiter,
             const int idx_column) {
  ofstream ofs(filename);
  int idx = 1;
  if (!ofs) {
    throw runtime_error("File not found: " + filename);
  }
  istringstream iss(header);
  string line;
  while (getline(iss, line)) {
    ofs << comments << " " << line << endl;
  }
  for (const auto &row : data) {
    if (idx_column) {
      ofs << idx << delimiter;
      idx++;
    }
    for (lu i = 0; i < row.size(); i++) {
      ofs << row[i];
      if (i < row.size() - 1) {
        ofs << delimiter;
      }
    }
    ofs << endl;
  }
}

template void savetxt<int>(const std::string &filename,
                           const std::vector<std::vector<int>> &data,
                           const std::string header, const std::string comments,
                           const char delimiter, const int idx_column);

template void savetxt<float>(const std::string &filename,
                             const std::vector<std::vector<float>> &data,
                             const std::string header,
                             const std::string comments, const char delimiter,
                             const int idx_column);

template void savetxt<double>(const std::string &filename,
                              const std::vector<std::vector<double>> &data,
                              const std::string header,
                              const std::string comments, const char delimiter,
                              const int idx_column);

double mse(const Eigen::VectorXf &x, const Eigen::VectorXf &y) {
  if (x.size() != y.size()) {
    throw runtime_error("Size mismatch.");
  }
  Eigen::VectorXd diff = x.cast<double>() - y.cast<double>();
  return diff.squaredNorm() / x.size();
}

}  // namespace util

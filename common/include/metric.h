#ifndef _ADPREDICTALGO_METRIC_
#define _ADPREIDCTALGO_METRIC_

namespace adPredictAlgo {

class Metric 
{
    public:
        // <pscore , label>
        typedef struct pair{
            pair(float s,int l):score(s),t_label(l) { }
            float score;
            int t_label;

            bool operator < (const struct pair &p) const {
                return score > p.score;
            }
        }pair_t;

    public:
        static double CalAUC(std::vector<pair_t> p) {
            long total_score = 0;
            long pos_num = 0;

            sort(p.begin(),p.end());
            size_t num = p.size();

            for(size_t i = 0;i < num;i++) {
                if(p[i].t_label == 1) {
                    total_score = total_score + num - i;
                    pos_num++;
                }
            }
            total_score = total_score - pos_num * (pos_num + 1)/2.0f;
            return (total_score * 1.0f / ((num - pos_num) * pos_num));
        }

        static double CalCOPC(const std::vector<pair_t> &p) {
            float score = 0.0;
            int label = 0;
            for(auto iter = p.begin();iter != p.end();iter++) {
                score += (*iter).score;
                label += (*iter).t_label;
            }
            return label * 1.0 / score;
        }


};//class Metric
}//namespace

#endif

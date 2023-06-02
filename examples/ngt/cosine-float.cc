
#include    "tann/ngt/index.h"

using namespace std;

int
main(int argc, char **argv) {
    string indexPath = "index";
    string objectFile = "./data/sift-dataset-5k.tsv";
    string queryFile = "./data/sift-query-3.tsv";
    // index construction
    try {
        tann::ngt::Property property;
        property.dimension = 128;
        property.objectType = tann::ngt::ObjectSpace::ObjectType::Float;
        property.distanceType = tann::ngt::Index::Property::DistanceType::DistanceTypeCosine;
        tann::ngt::Index::create(indexPath, property);
        tann::ngt::Index index(indexPath);
        ifstream is(objectFile);
        string line;
        while (getline(is, line)) {
            vector<float> obj;
            stringstream linestream(line);
            while (!linestream.eof()) {
                float value;
                linestream >> value;
                if (linestream.fail()) {
                    obj.clear();
                    break;
                }
                obj.push_back(value);
            }
            if (obj.empty()) {
                cerr << "An empty line or invalid value: " << line << endl;
                continue;
            }
            obj.resize(property.dimension);  // cut off additional data in the file.
            index.append(obj);
        }
        index.createIndex(16);
        index.save();
    } catch (tann::ngt::Exception &err) {
        cerr << "Error " << err.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error" << endl;
        return 1;
    }

    // nearest neighbor search
    try {
        tann::ngt::Index index(indexPath);
        tann::ngt::Property property;
        index.getProperty(property);
        ifstream is(queryFile);
        string line;
        while (getline(is, line)) {
            vector<float> query;
            {
                stringstream linestream(line);
                while (!linestream.eof()) {
                    float value;
                    linestream >> value;
                    query.push_back(value);
                }
                query.resize(property.dimension);
                cout << "Query : ";
                for (size_t i = 0; i < 5; i++) {
                    cout << query[i] << " ";
                }
                cout << "...";
            }
            tann::ngt::SearchQuery sc(query);
            tann::ngt::ObjectDistances objects;
            sc.setResults(&objects);
            sc.setSize(10);
            sc.setEpsilon(0.1);

            index.search(sc);
            cout << endl << "Rank\tID\tDistance" << std::showbase << endl;
            for (size_t i = 0; i < objects.size(); i++) {
                cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
                tann::ngt::ObjectSpace &objectSpace = index.getObjectSpace();
                float *object = static_cast<float *>(objectSpace.getObject(objects[i].id));
                for (size_t idx = 0; idx < 5; idx++) {
                    cout << object[idx] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    } catch (tann::ngt::Exception &err) {
        cerr << "Error " << err.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error" << endl;
        return 1;
    }

    return 0;
}


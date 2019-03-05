#include <fstream>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

struct SA_Node {
  uint32_t sa_nid;
  std::string name;
  uint32_t tensor_nid;
  uint32_t tensor_idx;
  std::vector<uint32_t> deps;
};

void LoadSaGraphFile(std::unordered_map<uint32_t, SA_Node>& sa_nodes) {
    std::ifstream ifs("thefile.txt");
    std::string line;
    while (std::getline(ifs, line)) {
        size_t next = 0, last = 0;
        std::string temp;
        next = line.find(",", last);
        //node.sa_nid = std::stoi(line.substr(last, next - last));
        uint32_t sa_nid = std::stoi(line.substr(last, next - last));
        sa_nodes[sa_nid].sa_nid = sa_nid;
        SA_Node& node = sa_nodes[sa_nid];
        last = next + 1;
        next = line.find(",", last);
        node.name = line.substr(last, next - last);
        last = next + 1;
        next = line.find(",", last);
        temp = line.substr(last, next - last);
        last = next + 1;
        next = line.find(",", last);
        if (temp[0] == 'N') {
            node.tensor_nid = -1;
            node.tensor_idx = -1;
        } else {
            node.tensor_nid = std::stoi(temp);
            temp = line.substr(last, next - last);
            node.tensor_idx = std::stoi(temp);
        }
        last = next + 1;
	while ((next = line.find(",", last)) != std::string::npos) {
	    node.deps.push_back(std::stoi(line.substr(last, next - last)));
	    last = next + 1;
	}
    }
}

NodeEntry CreateSwapEntry(const Op* swap_source_op) {
  NodePtr node = Node::Create();
  node->attrs.op = swap_source_op;
  node->attrs.name = "swap_entry";
  node->attrs.op->attr_parser(&(node->attrs));
  std::ostringstream os;
  os << "_SwapEntry_var";
  node->inputs.emplace_back(Symbol::CreateVariable("_SwapEntry_var").outputs[0]);
  return NodeEntry{std::move(node), 0, 0};
}

NodeEntry CreateSwapoutSink(const Op* swapout_sink_op) {
  NodePtr node = Node::Create();
  node->attrs.op = swapout_sink_op;
  node->attrs.name = "swapout_sink";
  node->attrs.op->attr_parser(&(node->attrs));
  return NodeEntry{std::move(node), 0, 0};
}

void CreateSwapOut(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                   const NodeEntry& swap_entry,
                   const NodeEntry& swapout_sink,
                   const Op* swapout_op,
                   std::unordered_map<uint32_t, NodeEntry>& swapouts) {
  for (const auto& kv: sa_nodes) {
    NodePtr node = Node::Create();
    node->attrs.op = swapout_op;
    node->attrs.name = "swapout";
    node->attrs.op->attr_parser(&(node->attrs));
    node->inputs.emplace_back(swap_entry);
    swapout_sink.node->control_deps.emplace_back(node);
    swapouts[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateSwapIn(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                  const NodeEntry& swap_entry,
                  const Op* swapin_op,
                  std::unordered_map<uint32_t, NodeEntry>& swapins) {
  for (const auto& kv: sa_nodes) {
    NodePtr node = Node::Create();
    node->attrs.op = swapin_op;
    node->attrs.name = "swapout";
    node->attrs.op->attr_parser(&(node->attrs));
    node->inputs.emplace_back(swap_entry);
    swapins[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateVariables(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                     const IndexedGraph& idx,
                     const NodeEntry& swap_entry,
                     std::unordered_map<uint32_t, NodeEntry>& variables) {
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (!idx[nid].source->is_variable()) continue;
    NodePtr node = nullptr;;
    // FIXME: This is very hacky. Any better way to get NodePtr ?
    for (uint32_t dep_nid = 0; nid < idx.num_nodes(); ++dep_nid) {
      for (uint32_t control_idx = 0; control_idx < idx[nid].control_deps.size();
           ++control_idx) {
        if (idx[dep_nid].control_deps[control_idx] == nid) {
          node = idx[dep_nid].source->control_deps[control_idx];
          break;
        }
      }
      if (node != nullptr) {
        break;
      }
      for (uint32_t input_idx = 0; input_idx < idx[dep_nid].inputs.size();
           ++input_idx) {
        if (idx[nid].inputs[input_idx].node_id == nid) {
          node = idx[nid].source->inputs[input_idx].node;
          break;
        }
      }
      if (node != nullptr) {
        break;
      }
    }
    CHECK(node != nullptr);
    LOG(INFO) << "Create variable " << node->attrs.name << std::endl;
    LOG(INFO) << "Create variable " << sa_nodes.at(nid).name << std::endl;
    CHECK(node->attrs.name == sa_nodes.at(nid).name);
    node->control_deps.emplace_back(swap_entry.node);
    variables[nid] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateModelNodes(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (idx[nid].source->is_variable()) continue;
    NodePtr new_node = Node::Create();
    new_node->attrs = idx[nid].source->attrs;
    LOG(INFO) << "Create node " << new_node->attrs.name << std::endl;
    LOG(INFO) << "Create node " << sa_nodes.at(nid).name << std::endl;
    CHECK(new_node->attrs.name == sa_nodes.at(nid).name);
    new_nodes[nid] = new_node;
  }
}

void ConnectPreSwapIn(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& variables) {
  for (auto& kv: swapins) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() > 0) continue;
    NodeEntry& entry = kv.second;
    variables[sa_nid].node->control_deps.emplace_back(entry.node);
  }
}

void ConnectAllSwapIn(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& swapouts,
                      std::unordered_map<uint32_t, NodeEntry>& variables,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  for (auto& kv: swapins) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() == 0) continue;
    NodeEntry& entry = kv.second;
    for (uint32_t dep_nid : sa_nodes.at(sa_nid).deps) {
      // Depend on another swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        entry.node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        entry.node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        entry.node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      // Depend on a model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        entry.node->control_deps.emplace_back(node_it->second);
      }
      CHECK(false);
    }
  }
}

void ConnectSwapOut(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                    const IndexedGraph& idx,
                    std::unordered_map<uint32_t, NodeEntry>& swapouts,
                    std::unordered_map<uint32_t, NodeEntry>& swapins,
                    std::unordered_map<uint32_t, NodeEntry>& variables,
                    std::unordered_map<uint32_t, NodePtr>& new_nodes) {

  for (auto& kv: swapouts) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() == 0) continue;
    NodeEntry& entry = kv.second;
    for (uint32_t dep_nid : sa_nodes.at(sa_nid).deps) {
      // Depend on another swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        entry.node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        entry.node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        entry.node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      // Depend on a model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        entry.node->control_deps.emplace_back(node_it->second);
      }
      CHECK(false);
    }
  }
}

void ConnectModelNodes(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                       const IndexedGraph& idx,
                       std::unordered_map<uint32_t, NodePtr>& new_nodes,
                       const std::unordered_map<uint32_t, NodeEntry>& swapouts,
                       const std::unordered_map<uint32_t, NodeEntry>& swapins,
                       std::unordered_map<uint32_t, NodeEntry>& variables) {
  for (auto& kv : new_nodes) {
    uint32_t sa_nid = kv.first;
    auto old_inode = idx[sa_nid];
    NodePtr new_node = kv.second;

    // Copy inputs.
    for (const IndexedGraph::NodeEntry& ientry : old_inode.inputs) {
      new_node->inputs.emplace_back(NodeEntry{new_nodes.at(ientry.node_id),
                                              ientry.index,
                                              ientry.version});
    }

    // Copy control nodes.
    const SA_Node& sa_node = sa_nodes.at(sa_nid);
    for (uint32_t dep_nid : sa_node.deps) {
      // Depend on another model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        new_node->control_deps.emplace_back(node_it->second);
      }

      // Depend on a swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        new_node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        new_node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        new_node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      CHECK(false);
    }
  }
}

Graph SA_LoadGraph(Graph src) {
  CHECK(src.attrs.count("swapout_op"))
      << "Need graph attribute \"swapout_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("swapin_op"))
      << "Need graph attribute \"swapin_op\" in SA_LoadGraph";
  const Op* swap_entry_op = Op::Get(src.GetAttr<std::string>("swap_entry_op"));
  const Op* swapout_sink_op = Op::Get(src.GetAttr<std::string>("swapout_sink_op"));
  const Op* swapin_op = Op::Get(src.GetAttr<std::string>("swapin_op"));
  const Op* swapout_op = Op::Get(src.GetAttr<std::string>("swapout_op"));
  const IndexedGraph& idx = src.indexed_graph();
  std::unordered_map<uint32_t, NodeEntry> swapouts;      // SA_ID -> swapout NodeEntry
  std::unordered_map<uint32_t, NodeEntry> swapins;       // SA_ID -> swapin NodeEntry
  std::unordered_map<uint32_t, SA_Node> sa_nodes;        // SA_ID -> SA_Node
  std::unordered_map<uint32_t, NodeEntry> variables;     // SA_ID -> new variable NodeEntry
  std::unordered_map<uint32_t, NodePtr> new_nodes;       // SA_ID -> new model NodeEntry

  // Create all the new swapout, swapin and nodes.
  // Connect all of them together.
  LoadSaGraphFile(sa_nodes);
  NodeEntry swap_entry = CreateSwapEntry(swap_entry_op);
  NodeEntry swapout_sink = CreateSwapoutSink(swapout_sink_op);
  CreateSwapOut(sa_nodes, swap_entry, swapout_sink, swapout_op, swapouts);
  CreateSwapIn(sa_nodes, swap_entry, swapin_op, swapins);
  CreateVariables(sa_nodes, idx, swap_entry, variables);
  CreateModelNodes(sa_nodes, idx, new_nodes);
  ConnectPreSwapIn(sa_nodes, idx, swapins, variables);
  ConnectAllSwapIn(sa_nodes, idx, swapins, swapouts, variables, new_nodes);
  ConnectSwapOut(sa_nodes, idx, swapouts, swapins, variables, new_nodes);
  ConnectModelNodes(sa_nodes, idx, new_nodes, swapouts, swapins, variables);

  // Create a new graph
  Graph ret;
  for (const NodeEntry& e : src.outputs) {
    CHECK(new_nodes.count(idx.node_id(e.node.get())) == 1);
    ret.outputs.emplace_back(NodeEntry{new_nodes[idx.node_id(e.node.get())],
                                       e.index,
                                       e.version});
  }
  ret.outputs.emplace_back(swapout_sink);
  return ret;
}

NNVM_REGISTER_PASS(SA_LoadGraph)
.describe("Load the new dataflow graph generated by the algorithms.")
.set_body(SA_LoadGraph)
.set_change_graph(true)
.depend_graph_attr("swap_entry_op")
.depend_graph_attr("swapout_sink_op")
.depend_graph_attr("swapin_op")
.depend_graph_attr("swapout_op");
}  // namespace
}  // namespace pass
}  // namespace nnvm

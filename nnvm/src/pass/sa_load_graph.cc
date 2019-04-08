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
  std::unordered_set<uint32_t> be_depended;
};

void LoadSAGraphFile(std::unordered_map<uint32_t, SA_Node>& sa_nodes) {
  std::cout << "LoadSAGraphFile" << std::endl;
  std::ifstream ifs("dataflow.rst");
  std::string line;
  while (std::getline(ifs, line)) {
    size_t next = 0, last = 0;
    next = line.find(",", last);
    uint32_t sa_nid = std::stoi(line.substr(last, next - last));
    sa_nodes[sa_nid].sa_nid = sa_nid;
    SA_Node& node = sa_nodes[sa_nid];
    last = next + 1;
    next = line.find(",", last);
    node.name = line.substr(last, next - last);
    last = next + 1;
    next = line.find(",", last);
    std::string nid = line.substr(last, next - last);
    last = next + 1;
    next = line.find(",", last);
    std::string idx = line.substr(last, next - last);
    if (nid[0] == 'N') {
      node.tensor_nid = -1;
      node.tensor_idx = -1;
    } else {
      node.tensor_nid = std::stoi(nid);
      node.tensor_idx = std::stoi(idx);
    }
    last = next + 1;
    while ((next = line.find(",", last)) != std::string::npos) {
      node.deps.push_back(std::stoi(line.substr(last, next - last)));
      last = next + 1;
    }
  }
  std::cout << "SA_Node count " << sa_nodes.size() << std::endl;
}

NodeEntry CreateSwapEntry(const Op* swap_source_op) {
  std::cout << "CreateSwapEntry" << std::endl;
  NodePtr node = Node::Create();
  node->attrs.op = swap_source_op;
  node->attrs.name = "swap_entry";
  //node->attrs.op->attr_parser(&(node->attrs));
  std::ostringstream os;
  os << "_SwapEntry_var";
  //node->inputs.emplace_back(Symbol::CreateVariable("_SwapEntry_var").outputs[0]);
  return NodeEntry{std::move(node), 0, 0};
}

NodeEntry CreateSwapoutSink(const Op* swapout_sink_op) {
  std::cout << "CreateSwapoutSink" << std::endl;
  NodePtr node = Node::Create();
  node->attrs.op = swapout_sink_op;
  node->attrs.name = "swapout_sink";
  //node->attrs.op->attr_parser(&(node->attrs));
  return NodeEntry{std::move(node), 0, 0};
}

void CreateSwapout(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                   const NodeEntry& swap_entry,
                   const NodeEntry& swapout_sink,
                   const Op* swapout_op,
                   std::unordered_map<uint32_t, NodeEntry>& swapouts) {
  std::cout << "CreateSwapout" << std::endl;
  for (const auto& kv: sa_nodes) {
    NodePtr node = Node::Create();
    node->attrs.op = swapout_op;
    node->attrs.name = "swapout";
    //node->attrs.op->attr_parser(&(node->attrs));
    //node->inputs.emplace_back(swap_entry);
    node->control_deps.emplace_back(swap_entry.node);
    swapout_sink.node->control_deps.emplace_back(node);
    swapouts[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                  const NodeEntry& swap_entry,
                  const Op* swapin_op,
                  std::unordered_map<uint32_t, NodeEntry>& swapins) {
  std::cout << "CreateSwapin" << std::endl;
  for (const auto& kv: sa_nodes) {
    NodePtr node = Node::Create();
    node->attrs.op = swapin_op;
    node->attrs.name = "swapout";
    //node->attrs.op->attr_parser(&(node->attrs));
    //node->inputs.emplace_back(swap_entry);
    node->control_deps.emplace_back(swap_entry.node);
    swapins[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateVariables(std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                     const IndexedGraph& idx,
                     const NodeEntry& swap_entry,
                     std::unordered_map<uint32_t, NodeEntry>& variables,
                     std::unordered_map<Node*, uint32_t>& nodeptr_to_old_nid) {
  std::cout << "CreateVariables" << std::endl;
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
        if (idx[dep_nid].inputs[input_idx].node_id == nid) {
          node = idx[dep_nid].source->inputs[input_idx].node;
          break;
        }
      }
      if (node != nullptr) {
        break;
      }
    }
    CHECK(node != nullptr);
    //LOG(INFO) << "Create variable " << node->attrs.name << std::endl;
    //LOG(INFO) << "Create variable " << sa_nodes.at(nid).name << std::endl;
    CHECK(node->attrs.name == sa_nodes.at(nid).name);
    if (node->attrs.name == "data") {
      std::cout << "Found data" << std::endl;
      variables[nid] = NodeEntry{std::move(node), 0, 0};
      swap_entry.node->inputs.emplace_back(variables[nid]);
    } else {
      node->control_deps.emplace_back(swap_entry.node);
      variables[nid] = NodeEntry{std::move(node), 0, 0};
    }
    for (const auto dep_nid : sa_nodes.at(nid).deps) {
      sa_nodes.at(dep_nid).be_depended.insert(nid);
    }
    nodeptr_to_old_nid[node.get()] = nid;
  }
}

void CreateModelNodes(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes,
                      std::unordered_map<Node*, uint32_t>& nodeptr_to_old_nid) {
  std::cout << "CreateModelNodes" << std::endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (idx[nid].source->is_variable()) continue;
    NodePtr new_node = Node::Create();
    new_node->attrs = idx[nid].source->attrs;
    //LOG(INFO) << "Create node " << new_node->attrs.name << std::endl;
    //LOG(INFO) << "Create node " << sa_nodes.at(nid).name << std::endl;
    CHECK(new_node->attrs.name == sa_nodes.at(nid).name);
    new_nodes[nid] = new_node;
    nodeptr_to_old_nid[new_node.get()] = nid;
  }
}

void ConnectPreSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& variables) {
  std::cout << "ConnectPreSwapin" << std::endl;
  for (auto& kv: swapins) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() > 0) continue;
    NodeEntry& entry = kv.second;
    for (const auto var_id : sa_nodes.at(sa_nid).be_depended) {
      variables.at(var_id).node->control_deps.emplace_back(entry.node);
    }
  }
}

void ConnectAllSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& swapouts,
                      std::unordered_map<uint32_t, NodeEntry>& variables,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  std::cout << "ConnectAllSwapin" << std::endl;
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

void ConnectSwapout(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                    const IndexedGraph& idx,
                    std::unordered_map<uint32_t, NodeEntry>& swapouts,
                    std::unordered_map<uint32_t, NodeEntry>& swapins,
                    std::unordered_map<uint32_t, NodeEntry>& variables,
                    std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  std::cout << "ConnectAllSwapout" << std::endl;
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
  std::cout << "ConnectModelNodes" << std::endl;
  for (auto& kv : new_nodes) {
    uint32_t sa_nid = kv.first;
    auto old_inode = idx[sa_nid];
    NodePtr new_node = kv.second;

    // Copy inputs.
    std::cout << "ConnectModeNode " << old_inode.source->attrs.name << std::endl;
    std::cout << "ConnectModeNode " << old_inode.source->attrs.op->name << std::endl;
    std::cout << "ConnectModeNode inputs " << old_inode.inputs.size() << std::endl;
    std::cout << "ConnectModeNode old_deps " << old_inode.control_deps.size() << std::endl;
    for (const IndexedGraph::NodeEntry& ientry : old_inode.inputs) {
      auto var_it = variables.find(ientry.node_id);
      if (var_it != variables.end()) {
        new_node->inputs.emplace_back(NodeEntry{var_it->second.node,
                                                ientry.index,
                                                ientry.version});
      } else {
        new_node->inputs.emplace_back(NodeEntry{new_nodes.at(ientry.node_id),
                                                ientry.index,
                                                ientry.version});
      }
    }

    std::vector<uint32_t> deps;
    // Copy control nodes.
    for (uint32_t dep_nid : old_inode.control_deps) {
      deps.push_back(dep_nid);
    }
    // Control deps from SA
    const SA_Node& sa_node = sa_nodes.at(sa_nid);
    for (const uint32_t dep_nid : sa_node.deps) {
      bool exist = false;
      for (const uint32_t  _dep_nid : deps) {
        if (_dep_nid == dep_nid) {
          exist = true;
          break;
        }
      }
      if (!exist) {
        deps.push_back(dep_nid);
      }
    }

    std::cout << "ConnectModeNode new_deps" << deps.size() << std::endl;
    for (uint32_t dep_nid : deps) {
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
  std::cout << "SA_LoadGraph" << std::endl;
  CHECK(src.attrs.count("swapout_op"))
      << "Need graph attribute \"swapout_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("swapin_op"))
      << "Need graph attribute \"swapin_op\" in SA_LoadGraph";
  const Op* swap_entry_op = Op::Get(src.GetAttr<std::string>("swap_entry_op"));
  const Op* swapout_sink_op = Op::Get(src.GetAttr<std::string>("swapout_sink_op"));
  const Op* swapin_op = Op::Get(src.GetAttr<std::string>("swapin_op"));
  const Op* swapout_op = Op::Get(src.GetAttr<std::string>("swapout_op"));
  const IndexedGraph& idx = src.indexed_graph();
  std::unordered_map<Node*, uint32_t> nodeptr_to_old_nid;
  std::unordered_map<uint32_t, NodeEntry> swapouts;       // SA_ID -> swapout NodeEntry
  std::unordered_map<uint32_t, NodeEntry> swapins;        // SA_ID -> swapin NodeEntry
  std::unordered_map<uint32_t, SA_Node> sa_nodes;         // SA_ID -> SA_Node
  std::unordered_map<uint32_t, NodeEntry> variables;      // SA_ID -> new variable NodeEntry
  std::unordered_map<uint32_t, NodePtr> new_nodes;        // SA_ID -> new model NodeEntry

  // Create all the new swapout, swapin and nodes.
  // Connect all of them together.
  LoadSAGraphFile(sa_nodes);
  NodeEntry swap_entry = CreateSwapEntry(swap_entry_op);
  NodeEntry swapout_sink = CreateSwapoutSink(swapout_sink_op);
  CreateSwapout(sa_nodes, swap_entry, swapout_sink, swapout_op, swapouts);
  CreateSwapin(sa_nodes, swap_entry, swapin_op, swapins);
  CreateVariables(sa_nodes, idx, swap_entry, variables, nodeptr_to_old_nid);
  CreateModelNodes(sa_nodes, idx, new_nodes, nodeptr_to_old_nid);
  ConnectPreSwapin(sa_nodes, idx, swapins, variables);
  ConnectAllSwapin(sa_nodes, idx, swapins, swapouts, variables, new_nodes);
  ConnectSwapout(sa_nodes, idx, swapouts, swapins, variables, new_nodes);
  ConnectModelNodes(sa_nodes, idx, new_nodes, swapouts, swapins, variables);

  // Create a new graph
  std::cout << "Create a new graph" << std::endl;
  Graph ret;
  size_t output_idx = 0;
  size_t num_forward_outputs = src.GetAttr<size_t>("num_forward_outputs");
  for (const NodeEntry& e : src.outputs) {
    CHECK(new_nodes.count(idx.node_id(e.node.get())) == 1);
    ret.outputs.emplace_back(NodeEntry{new_nodes[idx.node_id(e.node.get())],
                                       e.index,
                                       e.version});
    if (output_idx + 1 == num_forward_outputs) {
      ret.outputs.emplace_back(swapout_sink);
    }
    output_idx += 1;
  }

  // Update graph attributes.
  const auto& new_idx = ret.indexed_graph();
  NodeIdMap old_nids;
  EntryIdMap old_eids;
  std::cout << "nodeptr_to_old_nid " << nodeptr_to_old_nid.size() << std::endl;
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const auto it =
      nodeptr_to_old_nid.find(const_cast<Node*>(new_idx[nid].source));
    if (it == nodeptr_to_old_nid.end()) {
      continue;
    }
    const size_t old_nid = it->second;
    old_nids[nid] = old_nid;
    const size_t num_outputs = new_idx[nid].source->num_outputs();
    for (size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
      old_eids[new_idx.entry_id(nid, output_idx)] = idx.entry_id(old_nid,
                                                                 output_idx);
    }
  }
  ret.attrs["context"] = src.attrs.at("context");
  ret.attrs["device"] = src.attrs.at("device");
  ret.attrs["old_nids"] = std::make_shared<dmlc::any>(std::move(old_nids));
  ret.attrs["old_eids"] = std::make_shared<dmlc::any>(std::move(old_eids));
  ret.attrs["num_forward_inputs"] =
    std::make_shared<dmlc::any>(src.GetAttr<size_t>("num_forward_inputs"));
  ret.attrs["num_forward_outputs"] =
    std::make_shared<dmlc::any>(num_forward_outputs + 1);

  return ret;
}

NNVM_REGISTER_PASS(SA_LoadGraph)
.describe("Load the new dataflow graph generated by the algorithms.")
.set_body(SA_LoadGraph)
.set_change_graph(true)
.provide_graph_attr("num_forward_inputs")
.provide_graph_attr("num_forward_outputs")
.provide_graph_attr("old_nids")
.provide_graph_attr("old_eids")
.depend_graph_attr("context")
.depend_graph_attr("device")
.depend_graph_attr("num_forward_inputs")
.depend_graph_attr("num_forward_outputs")
.depend_graph_attr("swap_entry_op")
.depend_graph_attr("swapout_sink_op")
.depend_graph_attr("swapin_op")
.depend_graph_attr("swapout_op");
}  // namespace
}  // namespace pass
}  // namespace nnvm

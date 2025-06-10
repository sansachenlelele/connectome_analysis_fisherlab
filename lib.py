import numpy as np
import pandas as pd
import statistics as stats

import bokeh
import hvplot.pandas
import holoviews as hv

import bokeh.palettes
from bokeh.plotting import figure, show, output_notebook

import neuprint


# NOTE: one distance unit in the connectome is 8nm, need to multiply by 8/1000 to get microns


class syn_specs:
    def __init__(self, target_neuron: int, scale: str, conn_type: str, conn_id: str | int | None = None, rois=None, lable_res: str | None = None, top: int | None = None, primary_only: bool = True):
        """ Class for handeling synapses between a specified target neuron and a connecting neuron/instance/type 
            * target_neuron (int): neuprint bodyId of the neuron would you like to examine
            * scale (str): scale at which to examine synapses to/from the target neuron
                - 'neuron': anylize synapses between the target neuron and the given neuron
                - 'instance': anylize synapses between the target neuron and the given neuron instance (subtype)
                    * NOTE: must specify neuprint neuron instance (subtype) name as 'conn_id' argument
                - 'type': anylize synapses between the target neuron and given neuron type
                    * NOTE: must specify neuprint neuron type name as 'conn_id' argument
                - 'all': anylize synapses between the target neuron and all connecting neurons
            * conn_type (str) ['pre', 'post']: specify weather you are interested in pre or post synaptic connections between the target neuron and connecting neuron class
                - This is from the point of view of the target neuron
            * conn_id (str): neuprint identifier for connecting neuron body Id/instance/type
                - Leave as None if interested in all pre/post synapses
            * rois (list of str): return only synapses within the given ROIs, leave as None if interested in all synapses
            * lable_res (str): resolution at which to label synapses (options: 'neuron', 'instance', 'type')
            * top (int): number of neurons to visualize connections from/to
                - If left as None will return synapses from/to all neurons matching the query, otherwise returns synapses from/to specified number of neurons sorted by highest number of synapses
            * primary_only (bool): return only primary synapses of the given type (do NOT include synapses from non-primary ROIs)
        """
        assert scale in ['neuron', 'instance', 'type', 'all'], "Error: must specify scale of 'neuron', 'instance', 'type', or 'all'"
        if lable_res: assert lable_res in ['neuron', 'instance', 'type'], "Error: must specify lable resolution of 'neuron', 'instance', or 'type'"
        assert conn_type in ['pre', 'post'], "Error: must specify connection type of either pre or post"

        self.target_neuron = target_neuron
        self.scale = scale
        self.conn_id = conn_id
        self.conn_type = conn_type
        self.rois = rois
        self.top = top
        self.primary_only = primary_only

        if lable_res=='neuron':
            self.lable_res = 'bodyId'
        else:
            self.lable_res = lable_res
        
        if scale=='neuron':
            conn_cri = neuprint.NeuronCriteria(bodyId=conn_id)
            self.lable_res = 'bodyId'
        elif scale=='instance':
            conn_cri = neuprint.NeuronCriteria(instance=conn_id)
            if lable_res==None or lable_res=='type': self.lable_res = 'bodyId'
        elif scale=='type':
            conn_cri = neuprint.NeuronCriteria(type=conn_id)
            if lable_res==None: self.lable_res = 'instance'
        else:
            conn_cri = None
            if lable_res==None: self.lable_res = 'type'
        
        neuron_cri = neuprint.NeuronCriteria(bodyId=target_neuron)
        self.syn_cri = neuprint.SynapseCriteria(rois=rois, primary_only=primary_only)
        if conn_type == 'pre':
            self.pre_cri = conn_cri
            self.post_cri = neuron_cri
        else:
            self.pre_cri = neuron_cri
            self.post_cri = conn_cri
    
    def fetch_syn_conns(self):
        print(f"Fetching {self.conn_type}-synaptic connections...")
        try:
            conn_df = neuprint.fetch_synapse_connections(self.pre_cri, self.post_cri, self.syn_cri)
            neurons, _ = neuprint.fetch_neurons(conn_df['bodyId_'+self.conn_type].unique())
            conn_df = neuprint.utils.merge_neuron_properties(neurons, conn_df)
            self.conns = conn_df.sort_values(f"type_{self.conn_type}")
        except RuntimeError:
            print("No synapses match your source criteria")
            self.conns = None
        return self.conns
    
    def create_points(self, palett=None, loop_colors=True):
        conn_df = self.fetch_syn_conns()
        if type(conn_df) != pd.core.frame.DataFrame:
            print("No synapses match specifications, can't create points")
            exit(0)
        if self.top: 
            top_conns = conn_df[self.lable_res+'_'+self.conn_type].value_counts().head(self.top)
        else:
            top_conns = conn_df[self.lable_res+'_'+self.conn_type].value_counts()
        df_pal = bokeh.palettes.Plasma if self.conn_type=='pre' else bokeh.palettes.Viridis
        if not palett:
            lc = len(top_conns)
            if lc <= 9:
                palett = bokeh.palettes.RdPu[lc if lc>2 else 3] if self.conn_type=='pre' else bokeh.palettes.YlGn[lc if lc>2 else 3]
            if lc <= 11:
                palett = df_pal[lc if lc>2 else 3]
            elif lc > 100:
                palett = df_pal[256]
            else:
                if loop_colors:
                    palett = df_pal[11]
                else:
                    palett = bokeh.palettes.Iridescent23 if self.conn_type=='pre' else bokeh.palettes.TolRainbow23
        points = conn_df.query(f'{self.lable_res}_{self.conn_type} in @top_conns.index').copy()
        colors = (palett * (len(points) // len(palett) + 1))[:len(points)]
        points['color'] = points[self.lable_res+'_'+self.conn_type].map(dict(zip(top_conns.index, colors)))
        self.top_conns = top_conns
        self.points = points
        return self.top_conns, self.points


def skeleton_synapse_visualization(target_neuron: int, syn_classes, skeleton_color=bokeh.palettes.Inferno3[0], paletts=None, loop_colors:bool = True):
    """ Function returning a graphic of the skeleton of neuron specified by body_Id with the desired synapses plotted colored by pre/post synpase and neuron subtype
        * target_neuron (int): neuprint integer bodyId of the neuron would you like to examine
        * syn_classes (list of syn_spec objects): list of all classes of synapses you want to visualize
        * skeleton_color (bokeh palett): desired color descriptor for skeleton (default: black)
        * paletts (list of bokeh paletts): bokeh palett to use for each synapse class
            - NOTE: if provided, the number of paletts must match the number of synapse classes specified
        * loop_colors (bool): flag for weather or not to repeat colors over multiple neurons 
            - TIP: if you are plotting connections from less than ~100 neurons you should leave this as 'True' or the difference bewteen synapses of different neurons will be very hard to visually distingish 
            - NOTE: The maximum number of colors avaiable in the palettes is 256 so if you are plotting connections to more neurons than the colors will be repeated regardless of how you set this flag
    """
    # TODO: figure out how to generate color-coded lable_res legend
    # TODO: add option to visualize additional neuron skeletons
    if paletts: assert len(paletts) == len(syn_classes), "Error: must specify a palett for each synapse class"
    p = figure()
    p.y_range.flipped = True
    s = neuprint.skeleton.fetch_skeleton(target_neuron, format='pandas')
    s['bodyId'] = target_neuron
    s['color'] = skeleton_color
    s = s.merge(s, 'inner', left_on=['bodyId', 'link'], right_on=['bodyId', 'rowId'], suffixes=['_child', '_parent'])
    p.segment(x0='x_child', x1='x_parent', y0='z_child', y1='z_parent', color='color_child', source=s)
    top_conns = []
    for i, ss in enumerate(syn_classes):
        assert ss.target_neuron == target_neuron, "Error, all synapse classes must reference same target neuron"
        pal = paletts[i] if paletts else None
        top_conn, points = ss.create_points(palett=pal, loop_colors=loop_colors)
        top_conns.append(top_conn)
        p.scatter(points['x_pre'], points['z_pre'], color=points['color'])
    show(p)
    return top_conns


def fetch_connectivity(target_scale, conn_scale, conn_type, target_id, conn_id=None, rois=None, include_nonprimary=False):
    """ Fetch a connectivity matrix between specified neurons/subtypes/types avoiding over/under counting of synapses 
        * target_scale (str): indicates scale to analyze neuron(s) of interest on
            - 'neuron': normalize conections to/from a specific neuron 
                - NOTE: must specify neuprint neuron integer bodyId as 'target_id' argument
            - 'instance': normalize connections over an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'target_id' argument 
            - 'type': normalize connections over an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'target_id' argument 
        * conn_scale (str): indicates scale over which to analyze connections to/from target neuron(s)
            - 'neuron': normalize connections to/from a sprcific neuron
                - NOTE: must specify neuprint neuron integer bodyId as 'conn_id' argument
            - 'instance': nomalize connections to/from an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'conn_id' argument
            - 'type': normalize connections to/from an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'conn_id' argument
            - 'all': normalize connections to/from all pre/post synaptic neurons
        * conn_type (str): indicates weather to analyze inputs or outputs to/from a given neuron/instance/type
            - 'pre': normalize presynaptic connections (analyze relative contributions of inputs) 
            - 'post': normalize postsynaptic connections (analyze relative output strengths)
        * target_id (int or str): neuprint identifier for target neuron(s) ID/instance/type
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * conn_id (int, str, or None): neuprint identifier for connecting neuron(s) ID/instance/type
            - Leave as 'None' if you're interested in all connections to/from the target neuron(s)
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * rois (list of str): list of string identifiers for all ROIs from which to analyze connections from
            - Leave as None if interested in all connections bettween the specified neurons, regardless of location 
        * include_nonprimary (bool): flag indicating weather or not to include synapses from non-primary ROIs
            - NOTE: this should be set to 'True' if you are interested in synapses within a single glomeruli for example
    """
    assert target_scale in ['neuron', 'instance', 'type'], "Error: must specify target scale of 'neuron', 'instance', or 'type'"
    assert conn_scale in ['neuron', 'instance', 'type', 'all'], "Error: must specify connection scale of 'neuron', 'instance', 'type', or 'all'"
    assert conn_type in ['pre', 'post'], "Error: must specify connection type of 'pre' or 'post'"
    if target_scale == 'neuron':
        assert type(target_id) == int, "Error: must specify integer bodyId for target neuron"
        target_nc = neuprint.NeuronCriteria(bodyId=target_id)
    elif target_scale == 'instance':
        assert type(target_id) == str, "Error: must specify string neuprint instance name for target neuron subtype"
        target_nc = neuprint.NeuronCriteria(instance=target_id)
    else:
        assert type(target_id) == str, "Error: must specify string neuprint type name for connecting neuron type"
        target_nc = neuprint.NeuronCriteria(type=target_id)
    if conn_scale == 'neuron':
        assert type(conn_id) == int, "Error: must specify integer bodyId for connecting neuron"
        conn_nc = neuprint.NeuronCriteria(bodyId=conn_id)
    elif conn_scale == 'instance':
        assert type(conn_id) == str, "Error: must specify string neuprint instance name for connecting neuron subtype"
        conn_nc = neuprint.NeuronCriteria(instance=conn_id)
    elif conn_scale == 'type':
        assert type(conn_id) == str, "Error: must specify string neuprint type name for connecting neuron type"
        conn_nc = neuprint.NeuronCriteria(type=conn_id)
    else:
        conn_nc=None
    if conn_type == 'pre':
        pre_nc = conn_nc
        post_nc = target_nc
    else:
        pre_nc = target_nc
        post_nc = conn_nc
    neurons, conns = neuprint.fetch_adjacencies(pre_nc, post_nc, rois=rois, min_roi_weight=1, include_nonprimary=include_nonprimary)
    conns = neuprint.merge_neuron_properties(neurons, conns, ['type', 'instance'])
    conns.sort_values('weight', ascending=False, inplace=True)
    if not include_nonprimary:
        # manually remove any 'NotPrimary' synapses (even with include_nonprimary=False some are included!)
        conns = conns[conns['roi']!='NotPrimary']
    conns = conns[['bodyId_pre', 'instance_pre', 'type_pre', 'bodyId_post', 'instance_post', 'type_post', 'roi', 'weight']]
    return conns


def fetch_connectivity_multi(target_scale, target_id, conn_type, conn_specs, rois=None, include_nonprimary=False):
    """ Fetch a connectivity matrix between a given target neuron and multiple classes of pre/post synaptic neurons/subtypes/types avoiding over/under counting of synapses
        * NOTE: all connections must be either pre or post syanptic with respect to the specified target neuron (no mixing)
        * target_scale (str): indicates scale to analyze neuron(s) of interest on
            - 'neuron': normalize conections to/from a specific neuron 
                - NOTE: must specify neuprint neuron integer bodyId as 'target_id' argument
            - 'instance': normalize connections over an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'target_id' argument 
            - 'type': normalize connections over an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'target_id' argument 
         * target_id (int or str): neuprint identifier for target neuron(s) ID/instance/type
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
         * conn_type (str): indicates weather to analyzing inputs or outputs to/from a given neuron/instance/type
            - 'pre': normalize presynaptic connections (analyze relative contributions of inputs) 
            - 'post': normalize postsynaptic connections (analyze relative output strengths)
        * conn_specs: list of (conn_scale, conn_id) tuples for each connecting neuron class
            * conn_scale (str): indicates scale over which to analyze connections to/from target neuron(s)
                - 'neuron': normalize connections to/from a sprcific neuron
                    - NOTE: must specify neuprint neuron integer bodyId as 'conn_id' argument
                - 'instance': nomalize connections to/from an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                    - NOTE: must specify neuprint neuron instance (subtype) name as 'conn_id' argument
                - 'type': normalize connections to/from an entire type of neurons (ie 'PEN_b(PEN2)')
                    - NOTE: must specify neuprint neuron type name as 'conn_id' argument
                - 'all': normalize connections to/from all pre/post synaptic neurons
            * conn_id (int, str, or None): neuprint identifier for connecting neuron(s) ID/instance/type
                - Leave as 'None' if you're interested in all connections to/from the target neuron(s)
                - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * rois (list of str): list of string identifiers for all ROIs from which to analyze connections from
            - Leave as None if interested in all connections bettween the specified neurons, regardless of location 
        * include_nonprimary (bool): flag indicating weather or not to include synapses from non-primary ROIs
            - NOTE: this should be set to 'True' if you are interested in synapses within a single glomeruli for example
    """
    conns = []
    for conn_class in conn_specs:
        c = fetch_connectivity(target_scale=target_scale, conn_scale=conn_class[0], conn_type=conn_type, target_id=target_id, conn_id=conn_class[1], rois=rois, include_nonprimary=include_nonprimary)
        conns.append(c)
    conns = pd.concat(conns)
    conns.sort_values('weight', ascending=False, inplace=True)
    return conns


def get_synapse_cnt_stats(target_scale, conn_scale, conn_type, target_id, conn_id=None, rois=None, v=False):
    """ Get average number of synapses between individual neurons of a target neuron class and specified connecting neuron class
        * target_scale (str): indicates scale to analyze neuron(s) of interest on
            - 'neuron': normalize conections to/from a specific neuron 
                - NOTE: must specify neuprint neuron integer bodyId as 'target_id' argument
            - 'instance': normalize connections over an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'target_id' argument 
            - 'type': normalize connections over an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'target_id' argument 
        * conn_scale (str): indicates scale over which to analyze connections to/from target neuron(s)
            - 'instance': nomalize connections to/from an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'conn_id' argument
            - 'type': normalize connections to/from an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'conn_id' argument
            - 'all': normalize connections to/from all pre/post synaptic neurons
        * conn_type (str): indicates weather to analyzing inputs or outputs to/from a given neuron/instance/type
            - 'pre': normalize presynaptic connections (analyze relative contributions of inputs) 
            - 'post': normalize postsynaptic connections (analyze relative output strengths)
        * target_id (int or str): neuprint identifier for target neuron(s) ID/instance/type
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * conn_id (int, str, or None): neuprint identifier for connecting neuron(s) ID/instance/type
            - Leave as 'None' if you're interested in all connections to/from the target neuron(s)
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * rois (list of str): list of string identifiers for all ROIs from which to analyze connections from
            - Leave as None if interested in all connections bettween the specified neurons, regardless of location
        * v (bool): flag indicating weather or not to print connection statistics (default: False)
    """
    conns = fetch_connectivity(target_scale, conn_scale, conn_type, target_id, conn_id=conn_id, rois=rois)
    num_conns = len(conns['weight'])
    tot_syn = sum(conns['weight'])
    mean = (tot_syn / num_conns) if num_conns>0 else 0
    sd = stats.stdev(conns['weight']) if num_conns>1 else 0
    if v:
        print(f"Synapse count statistics for {target_id if conn_type=='post' else conn_id} => {target_id if conn_type=='pre' else conn_id} connections:")
        print(f" * Number connections:\t{num_conns}\n * Total synapse count:\t{tot_syn}\n * Mean synapse count:\t{mean:.02f}\n * Standard deviation:\t{sd:.02f}")
    return tot_syn, mean, sd, num_conns


def normalize_connectivity(target_scale, conn_scale, conn_type, target_id, conn_id=None, rois=None, norm_mode='syn_cnt'):
    """ Normalize a connectivity matrix between specified neurons/subtypes/types 
        * target_scale (str): indicates scale to analyze neuron(s) of interest on
            - 'neuron': normalize conections to/from a specific neuron 
                - NOTE: must specify neuprint neuron integer bodyId as 'target_id' argument
            - 'instance': normalize connections over an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'target_id' argument 
            - 'type': normalize connections over an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'target_id' argument 
        * conn_scale (str): indicates scale over which to analyze connections to/from target neuron(s)
            - 'instance': nomalize connections to/from an entire instance (subtype) of neurons (ie 'PEN_b(PB06b)_L4')
                - NOTE: must specify neuprint neuron instance (subtype) name as 'conn_id' argument
            - 'type': normalize connections to/from an entire type of neurons (ie 'PEN_b(PEN2)')
                - NOTE: must specify neuprint neuron type name as 'conn_id' argument
            - 'all': normalize connections to/from all pre/post synaptic neurons
        * conn_type (str): indicates weather to analyzing inputs or outputs to/from a given neuron/instance/type
            - 'pre': normalize presynaptic connections (analyze relative contributions of inputs) 
            - 'post': normalize postsynaptic connections (analyze relative output strengths)
        * target_id (int or str): neuprint identifier for target neuron(s) ID/instance/type
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * conn_id (int, str, or None): neuprint identifier for connecting neuron(s) ID/instance/type
            - Leave as 'None' if you're interested in all connections to/from the target neuron(s)
            - NOTE: nust exactly match neuron's identifier in the neuprint database including capatilization
        * rois (list of str): list of string identifiers for all ROIs from which to analyze connections from
            - Leave as None if interested in all connections bettween the specified neurons, regardless of location 
        * norm_mode (str): indicates the method of normalization to be preformed
            - 'syn_cnt': normalize connection strength between target neuron/instance/type and connection neuron/instance/type by average number of connections between target neuron/instance/type and connection neuron/instance/type (ignoring cell counts)
            - 'syn_tot' : normalize connection strength between target neuron/instance/type and connection neuron/instance/type by average total number of synapses to/from target neuron/instance/type (ignoring cell counts) 
            - 'cell_cnt': normalize connection strength between target neuron/instance/type and connection neuron/instance/type by average number of target neuron/instance/type neurons connecting to (pre/post) connection neuron/instance/type neurons (ignoring synapse counts)
            - 'cell_tot': normalize connection strength between target neuron/instance/type and connection neuron/instance/type by total number of neurons connecting to target neuron/instance/type neurons (ignoring synapse counts)
    """
    assert norm_mode in ['syn_cnt', 'syn_tot', 'cell_cnt', 'cell_tot'], "Error: must specify norm mode of 'syn_cnt', 'syn_tot', 'cell_cnt', or 'cell_tot'"
    if conn_scale == 'neuron':
        print("Cannot normalize connections on the scale of individual neurons. Nothing to do.")
        return None
    target_type = ('pre' if conn_type=='post' else 'post')
    ts_id = target_scale + '_' + target_type
    cs_id = conn_scale + '_' + conn_type
    conns = fetch_connectivity(target_scale, conn_scale, conn_type, target_id, conn_id, rois)
    if norm_mode == 'syn_cnt':
        avg_syn_cnt = sum(conns['weight']) / len(conns['weight'])
        conns['norm_syn_cnt'] = conns['weight'] / avg_syn_cnt
        conns['syn_cnt'] = conns['weight']
        conns = conns[['bodyId_pre', 'instance_pre', 'type_pre', 'bodyId_post', 'instance_post', 'type_post', 'roi', 'syn_cnt', 'norm_syn_cnt']]
    elif norm_mode == 'cell_cnt':
        cell_cnts = {}
        target_ids = conns['bodyId_'+target_type].unique()
        for bid in target_ids:
            cell_cnts[bid] = len(conns[conns['bodyId_'+target_type]==bid])
        avg_cnt = sum(cell_cnts.values()) / len(cell_cnts)
        cell_data = {'bodyId_'+target_type: cell_cnts.keys(), ts_id: [target_id]*len(cell_cnts), cs_id: [conn_id]*len(cell_cnts), conn_scale+'_cell_cnt_'+conn_type: cell_cnts.values(), 'norm_'+conn_scale+'_cell_cnt_'+conn_type: [x/avg_cnt for x in cell_cnts.values()] }
        conns = pd.DataFrame(cell_data)
        conns.sort_values(conn_scale+'_cell_cnt_'+conn_type, ascending=False, inplace=True)
    else:
        tot_conns = fetch_connectivity(target_scale=target_scale, conn_scale='all', conn_type=conn_type, target_id=target_id, conn_id=None, rois=None)
        if norm_mode == 'syn_tot':
            # calculate average pre/post synapse number over all neurons of target instance/type 
            avg_tot_syn_cnt = sum(tot_conns['weight']) / len(tot_conns['weight'])
            # normalize synapse count in conn table 
            conns['global_norm_syn_cnt'] = conns['weight'] / avg_tot_syn_cnt
            conns['syn_cnt'] = conns['weight']
            conns = conns[['bodyId_pre', 'instance_pre', 'type_pre', 'bodyId_post', 'instance_post', 'type_post', 'roi', 'syn_cnt', 'global_norm_syn_cnt']]
        else:
            # calaculate avarage pre/post synaptic cell count over all target instance/type
            # calc norm connection strength 
            target_ids = tot_conns['bodyId_'+target_type].unique()
            glob_cell_cnts = {}
            cell_cnts = {}
            for bid in target_ids:
                glob_cell_cnts[bid] = len(tot_conns[tot_conns['bodyId_'+target_type]==bid])
                cell_cnts[bid] = len(conns[conns['bodyId_'+target_type]==bid])
            glob_avg_cell_cnt = sum(glob_cell_cnts.values()) / len(glob_cell_cnts.values())
            cell_data = { 'bodyId_'+target_type: glob_cell_cnts.keys(), ts_id: [target_id]*len(glob_cell_cnts), 'tot_cell_cnt_'+conn_type: glob_cell_cnts.values(), cs_id: [conn_id]*len(glob_cell_cnts), conn_scale+'_cell_cnt_'+conn_type: cell_cnts.values(), 'norm_'+conn_scale+'_cell_cnt_'+conn_type: [x/glob_avg_cell_cnt for x in cell_cnts.values()] }
            conns = pd.DataFrame(cell_data)
            conns.sort_values(conn_scale+'_cell_cnt_'+conn_type, ascending=False, inplace=True)
    return conns


def normalize_connectivity_multi():
    # TODO: implement
    pass


def visualize_conn(conn_df, pre_scale, post_scale, sort_by='type', weight_col='weight', height=500, width=700, x_ax_rot=60):
    """ Function to plot connectivity dataframe as a heatmap
        * conn_df (pandas DataFrame object): connectivity table to plot
        * pre_scale (str): scale at which to group presynaptic neurons (plotted along the y-axis)
            - 'neuron': plot individual neurons labled by integer neuprint bodyId
            - 'instance': plot neurons grouped by instance labled by neuprint instance string
            - 'type': plot neurons grouped by type labled by neuprint type string
        * post_scale (str): scale at which to group postsynaptic neurons (plotted along the x-axis)
            - 'neuron': plot individual neurons labled by integer neuprint bodyId
            - 'instance': plot neurons grouped by instance labled by neuprint instance string
            - 'type': plot neurons grouped by type labled by neuprint type string
        * sort_by (str): desired ordering of neurons, options: ['instance', 'type'] 
        * weight_col (str): label of the column containing the weights to be plotted
        * height (int): desired hight of the plot
        * width (int): desired width of the plot
        * x_ax_rot (int): desired degree rotation of the x-axis lables
    """
    if pre_scale=='neuron': pre_scale = 'bodyId'
    if post_scale=='neuron': post_scale = 'bodyId'
    conn_mx = neuprint.connection_table_to_matrix(conn_df, group_cols=(pre_scale+'_pre', post_scale+'_post'), weight_col=weight_col, sort_by=sort_by)
    conn_mx.index = conn_mx.index.astype(str)
    conn_mx.columns = conn_mx.columns.astype(str)
    return conn_mx.hvplot.heatmap(height=height, width=width, xaxis='top').opts(xrotation=x_ax_rot)

